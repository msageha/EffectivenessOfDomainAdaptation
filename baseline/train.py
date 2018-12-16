import argparse
from collections import defaultdict
import json
import numpy as np
import torch.optim as optim
import os
import pandas as pd
from pprint import pprint
import torch
import torch.nn as nn
from tqdm import tqdm

from loader import WordVector, load_datasets, split
from model import BiLSTM

# init model
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Embedding') == -1):
        nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

def create_arg_parser():
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--type', dest='dataset_type', required=True, choices=['intra', 'inter'], help='dataset: "intra" or "inter"')
    parser.add_argument('--epochs', '-e', dest='max_epoch', type=int, default=10, help='max epoch')
    parser.add_argument('--emb_type', dest='emb_type', required=True, choices=['Word2Vec', 'FastText', 'ELMo', 'Random'], help='word embedding type')
    parser.add_argument('--emb_path', dest='emb_path',  help='word embedding path')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1, help='GPU ID for execution')
    parser.add_argument('--batch', '-b', dest='batch_size', type=int, default=32, help='mini batch size')
    parser.add_argument('--case', '-c', dest='case', type=str, required=True, choices=['ga', 'o', 'ni'], help='target "case" type')
    parser.add_argument('--media', '-m', dest='media', nargs='+', type=str, default=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], choices=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], help='training media type')
    parser.add_argument('--dump_dir', dest='dump_dir', type=str, required=True, help='model dump directory path')
    return parser

def initialize_model(gpu, vocab_size, v_vec):
    emb_dim = 200
    h_dim = 200
    class_num = 2
    is_gpu = True
    if gpu == -1:
        is_gpu = False
    bilstm = BiLSTM(emb_dim, h_dim, class_num, vocab_size, is_gpu, v_vec)
    if is_gpu:
        bilstm = bilstm.cuda()

    for m in bilstm.modules():
        print(m.__class__.__name__)
        weights_init(m)

    return bilstm

def dump_dic(dic, dump_dir, file_name):
    os.makedirs(f'./{dump_dir}/', exist_ok=True)
    with open(f'./{dump_dir}/{file_name}', 'w') as f:
        json.dump(dic, f, indent=2)

def translate_df_tensor(df_list, keys, argsort_index, gpu_id):
    vec = [np.array(i[keys], dtype=np.int) for i in df_list]
    vec = np.array(vec)[argsort_index]
    vec = [torch.tensor(i) for i in vec]
    vec = nn.utils.rnn.pad_sequence(vec, batch_first=True, padding_value=0)
    if gpu_id >= 0:
        vec = vec.cuda()
    return vec

def translate_batch(batch, gpu, case):
    x = batch[:, 0]
    y = batch[:, 1]
    files = batch[:, 2]
    batchsize = len(batch)
    #0 paddingするために，長さで降順にソートする．
    argsort_index = np.array([i.shape[0] for i in x]).argsort()[::-1]
    max_length = x[argsort_index[0]].shape[0]
    x_wordemb = translate_df_tensor(x, ['単語ID'], argsort_index, gpu)
    x_wordemb = x_wordemb.reshape(batchsize, -1)
    x_feature_emb_list = []
    for i in range(6):
        x_feature_emb = translate_df_tensor(x, [f'形態素{i}'], argsort_index, gpu)
        x_feature_emb = x_feature_emb.reshape(batchsize, -1)
        x_feature_emb_list.append(x_feature_emb)
    x_feature = translate_df_tensor(x, ['n単語目', 'n文節目','is主辞', 'is_target_verb', '述語からの距離'], argsort_index, gpu)
    x = [x_wordemb, x_feature_emb_list, x_feature]

    y = translate_df_tensor(y, [case], argsort_index, -1)
    y = y.reshape(batchsize)
    y = torch.eye(max_length, dtype=torch.long)[y]
    if args.gpu >= 0:
        y = y.cuda()

    files = files[argsort_index]
    return x, y, files

def train(trains, vals, bilstm, args):
    print('--- start training ---')
    epochs = args.max_epoch
    lr = 0.001 #学習係数
    results = {}
    optimizer = optim.Adam(bilstm.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs):
        N = len(trains)
        perm = np.random.permutation(N)
        running_loss = 0.0
        running_correct = 0
        bilstm.train()
        for i in tqdm(range(0, N, args.batch_size)):
            bilstm.zero_grad()
            optimizer.zero_grad()
            batch = trains[perm[i:i+args.batch_size]]
            x, y, _ = translate_batch(batch, args.gpu, args.case)
            batchsize = len(batch)

            out = bilstm.forward(x)
            out = torch.cat((out[:, :, 0].reshape(batchsize, 1, -1), out[:, :, 1].reshape(batchsize, 1, -1)), dim=1)
            pred = out.argmax(dim=2)[:, 1].cpu()
            running_correct += pred.eq(y.argmax(dim=1)).sum().item()

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[epoch: {epoch},\titer: {i+1}]\tloss: {running_loss/100}\tacc: {running_correct/100}')
                running_loss = 0.0
                running_correct = 0
        _results = test(vals, bilstm, args)
        results[epoch] = _results
        save_model(epoch, bilstm, args.dump_dir, args.gpu)
    dump_dic(results, args.dump_dir, 'training_logs.json')
    best_epochs = defaultdict(lambda: defaultdict(float))
    for epoch in results:
        for domain in sorted(results[epoch].keys()):
            if results[epoch][domain]['acc'] > best_epochs[domain]['acc']:
                best_epochs[domain]['acc'] = results[epoch][domain]['acc']
                best_epochs[domain]['epoch'] = epoch
    dump_dic(best_epochs, args.dump_dir, 'training_result.json')
    print('--- finish training ---\n--- best epochs for each domain ---')
    for domain in sorted(best_epochs.keys()):
        print(f'{domain} [epoch: {best_epochs[domain]["epoch"]}]\tacc: {best_epochs[domain]["acc"]}')

def test(tests, bilstm, args):
    bilstm.eval()
    results = defaultdict(lambda: defaultdict(float))
    criterion = nn.CrossEntropyLoss()
    N = len(tests)
    for i in tqdm(range(0, N, args.batch_size), mininterval=5):
        batch = tests[i:i+args.batch_size]
        batchsize = len(batch)
        x, y, files = translate_batch(batch, args.gpu, args.case)

        out = bilstm.forward(x)
        out = torch.cat((out[:, :, 0].reshape(batchsize, 1, -1), out[:, :, 1].reshape(batchsize, 1, -1)), dim=1)
        pred = out.argmax(dim=2)[:, 1].cpu()
        for i, file in enumerate(files):
            correct = pred[i].eq(y[i].argmax()).item()
            domain = return_file_domain(file)
            results[domain]['correct'] += correct
            results[domain]['samples'] += 1
            loss = criterion(out[i].reshape(1, 2, 85), y[i].reshape(1, 85))
            results[domain]['loss'] += loss
    all_samples = 0
    all_correct = 0
    all_loss = 0
    for domain in args.media:
        results['All']['loss'] += results[domain]['loss']
        results['All']['samples'] += results[domain]['samples']
        results['All']['correct'] += results[domain]['correct']
        results[domain]['loss'] /= results[domain]['samples']
        results[domain]['acc'] = results[domain]['correct']/results[domain]['samples']
    results['All']['loss'] /= results['All']['samples']
    results['All']['acc'] = results['All']['correct']/results['All']['samples']
    print(f'[epoch: {epoch}]')
    for domain in sorted(results.keys()):
        pprint(dict(results[domain]))
    return results

def return_file_domain(file):
    domain_dict = {'PM':'雑誌','PN':'新聞', 'OW':'白書', 'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'PB':'書籍'}
    for domain in domain_dict:
        if domain in file:
            return domain

def save_model(epoch, bilstm, dump_dir, gpu):
    print('--- save model ---')
    os.makedirs(f'./{dump_dir}/model/', exist_ok=True)
    bilstm.cpu()
    torch.save(bilstm.state_dict(), f'./{dump_dir}/model/{epoch}.pkl')
    if gpu >= 0:
        bilstm.cuda()

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    is_bin = True
    if args.emb_type == 'Random' or args.emb_type == 'ELMo':
        is_bin = False
    wv = WordVector(args.emb_path, is_bin)
    is_intra = True
    if args.dataset_type == 'inter':
        is_intra = False
    datasets = load_datasets(wv, is_intra, args.media)
    trains, vals, tests = split(datasets)
    args.__dict__['trains_size'] = len(trains)
    args.__dict__['vals_size'] = len(vals)

    bilstm = initialize_model(args.gpu, vocab_size=len(wv.index2word), v_vec= wv.vectors)
    dump_dic(args.__dict__, args.dump_dir, 'args.json')
    train(trains, vals, bilstm, args)
    # train_loader = data.DataLoader(trains, batch_size=16, shuffle=True)
    # vals_loader = data.DataLoader(vals, batch_size=16, shuffle=True)

if __name__ == '__main__':
    main()
# def load_model(epoch, bilstm, domain):
#     print('___load_model___')
#     bilstm.load_state_dict(torch.load(f'./model_{domain}/{epoch}.pkl'))
