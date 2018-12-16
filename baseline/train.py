import argparse
from collections import defaultdict
import numpy as np
import torch.optim as optim
import os
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

from loader import WordVector, load_datasets, split
from model import BiLSTM

domain_dict = {'PM':'雑誌','PN':'新聞', 'OW':'白書', 'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'PB':'書籍'}

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
    return parser

def initialize_model(gpu, v_vec):
    emb_dim = 200
    h_dim = 200
    class_num = 2
    is_gpu = True
    if gpu == -1:
        is_gpu = False
    vocab_size = len(wv.index2word)
    bilstm = BiLSTM(emb_dim, h_dim, class_num, vocab_size, is_gpu, v_vec)
    if is_gpu:
        bilstm = bilstm.cuda()

    for m in bilstm.modules():
        print(m.__class__.__name__)
        weights_init(m)

    return bilstm

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    is_bin = True
    if args.emb_type == 'Random' or args.emb_type == 'ELMo':
        is_bin = False
    wv = WordVector(args.emb_path, is_bin)
    is_intra = True
    if args.type == 'inter':
        is_intra = False
    datasets = load_datasets(wv, is_intra, args.media)
    trains, vals, tests = split(datasets)

    bilstm = initialize_model(args.gpu, wv.vectors)

    train(trains, vals, bilstm, args)
    # train_loader = data.DataLoader(trains, batch_size=16, shuffle=True)
    # vals_loader = data.DataLoader(vals, batch_size=16, shuffle=True)

def translate_batch(df_list, keys, argsort_index, gpu_id):
    vec = [np.array(i[keys], dtype=np.int) for i in df_list]
    vec = np.array(vec)[argsort_index]
    vec = [torch.tensor(i) for i in vec]
    vec = nn.utils.rnn.pdd_sequence(vec, batch_first=True, padding_value=0)
    if gpu_id >= 0:
        vec = vec.cuda()
    return vec

def train(trains, vals, bilstm, args):
    case = args.case
    batchsize = args.batch
    epochs = args.epochs
    lr = 0.001 #学習係数

    optimizer = optim.Adam(bilstm.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs):
        N = len(trains)
        perm = np.random.permutation(N)
        running_loss = 0.0
        bilstm.train()
        for i in tqdm(range(0, N, batchsize)):
            optimizer.zero_grad()
            batch = trains[perm[i:i+batchsize]]

            x = batch[:, 0]
            y = batch[:, 1]
            #0 paddingするために，長さで降順にソートする．
            argsort_index = np.array([i.shape[0] for i in x]).argsort()[::-1]
            x_wordemb = translate_batch(x, ['単語ID'], argsort_index, args.gpu)
            x_wordemb = [np.array(i['単語ID'], dtype=np.int) for i in x]
            x_featureemb0 = [np.array(i['形態素0'], dtype=np.int) for i in x]
            x_featureemb1 = [np.array(i['形態素1'], dtype=np.int) for i in x]
            x_featureemb2 = [np.array(i['形態素2'], dtype=np.int) for i in x]
            x_featureemb3 = [np.array(i['形態素3'], dtype=np.int) for i in x]
            x_featureemb4 = [np.array(i['形態素4'], dtype=np.int) for i in x]
            x_featureemb5 = [np.array(i['形態素5'], dtype=np.int) for i in x]
            x_feature = [np.array(i[['n単語目', 'n文節目','is主辞', 'is_target_verb', '述語からの距離']], dtype=np.int32) for i in x]

            x_wordemb = np.array(x_wordemb)[argsort_index]
            x_featureemb0 = np.array(x_featureemb0)[argsort_index]
            x_featureemb1 = np.array(x_featureemb1)[argsort_index]
            x_featureemb2 = np.array(x_featureemb2)[argsort_index]
            x_featureemb3 = np.array(x_featureemb3)[argsort_index]
            x_featureemb4 = np.array(x_featureemb4)[argsort_index]
            x_featureemb5 = np.array(x_featureemb5)[argsort_index]
            x_feature = np.array(x_feature)[argsort_index]

            x_wordemb = [torch.tensor(i) for i in x_wordemb]
            x_featureemb0 = [torch.tensor(i) for i in x_featureemb0]
            x_featureemb1 = [torch.tensor(i) for i in x_featureemb1]
            x_featureemb2 = [torch.tensor(i) for i in x_featureemb2]
            x_featureemb3 = [torch.tensor(i) for i in x_featureemb3]
            x_featureemb4 = [torch.tensor(i) for i in x_featureemb4]
            x_featureemb5 = [torch.tensor(i) for i in x_featureemb5]
            x_feature = [torch.tensor(i) for i in x_feature]

            x_wordemb = nn.utils.rnn.pad_sequence(x_wordemb, batch_first=True, padding_value=0)
            x_featureemb0 = nn.utils.rnn.pad_sequence(x_featureemb0, batch_first=True, padding_value=0)
            x_featureemb1 = nn.utils.rnn.pad_sequence(x_featureemb1, batch_first=True, padding_value=0)
            x_featureemb2 = nn.utils.rnn.pad_sequence(x_featureemb2, batch_first=True, padding_value=0)
            x_featureemb3 = nn.utils.rnn.pad_sequence(x_featureemb3, batch_first=True, padding_value=0)
            x_featureemb4 = nn.utils.rnn.pad_sequence(x_featureemb4, batch_first=True, padding_value=0)
            x_featureemb5 = nn.utils.rnn.pad_sequence(x_featureemb5, batch_first=True, padding_value=0)
            x_feature = nn.utils.rnn.pad_sequence(x_feature, batch_first=True, padding_value=0)

            y = list(y[argsort_index])
            for i, val in enumerate(y):
                index = val[case]
                y[i] = np.zeros(x_wordemb.shape[1])
                y[i][index] = 1

            y = torch.tensor(y, dtype=torch.long)

            if gpu:
                x_wordemb = x_wordemb.cuda()
                x_featureemb0 = x_featureemb0.cuda()
                x_featureemb1 = x_featureemb1.cuda()
                x_featureemb2 = x_featureemb2.cuda()
                x_featureemb3 = x_featureemb3.cuda()
                x_featureemb4 = x_featureemb4.cuda()
                x_featureemb5 = x_featureemb5.cuda()
                x_feature = x_feature.cuda()
                y = y.cuda()

            out = bilstm.forward(x_wordemb, x_featureemb0, x_featureemb1, x_featureemb2, x_featureemb3, x_featureemb4, x_featureemb5, x_feature)

            out = torch.cat((out[:, :, 0].reshape(x_wordemb.shape[0], 1, -1), out[:, :, 1].reshape(x_wordemb.shape[0], 1, -1)), 1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if i % 300 == 299:    # print every 300 mini-batches
                print(f'[{epoch}, {i+1}] loss: {running_loss/300}')
                running_loss = 0.0


        N = len(vals)
        perm = np.random.permutation(N)
        running_loss = 0.0
        bilstm.eval()
        correct = 0
        for i in tqdm(range(0, N, batchsize)):
            optimizer.zero_grad()
            batch = vals[perm[i:i+batchsize]]

            x = batch[:, 0]
            y = batch[:, 1]
            argsort_index = np.array([i.shape[0] for i in x]).argsort()[::-1]
            x_wordemb = [np.array(i['単語ID'], dtype=np.int) for i in x]
            x_featureemb0 = [np.array(i['形態素0'], dtype=np.int) for i in x]
            x_featureemb1 = [np.array(i['形態素1'], dtype=np.int) for i in x]
            x_featureemb2 = [np.array(i['形態素2'], dtype=np.int) for i in x]
            x_featureemb3 = [np.array(i['形態素3'], dtype=np.int) for i in x]
            x_featureemb4 = [np.array(i['形態素4'], dtype=np.int) for i in x]
            x_featureemb5 = [np.array(i['形態素5'], dtype=np.int) for i in x]
            x_feature = [np.array(i[['n単語目', 'n文節目','is主辞', 'is_target_verb', '述語からの距離']], dtype=np.int32) for i in x]

            x_wordemb = np.array(x_wordemb)[argsort_index]
            x_featureemb0 = np.array(x_featureemb0)[argsort_index]
            x_featureemb1 = np.array(x_featureemb1)[argsort_index]
            x_featureemb2 = np.array(x_featureemb2)[argsort_index]
            x_featureemb3 = np.array(x_featureemb3)[argsort_index]
            x_featureemb4 = np.array(x_featureemb4)[argsort_index]
            x_featureemb5 = np.array(x_featureemb5)[argsort_index]
            x_feature = np.array(x_feature)[argsort_index]

            x_wordemb = [torch.tensor(i) for i in x_wordemb]
            x_featureemb0 = [torch.tensor(i) for i in x_featureemb0]
            x_featureemb1 = [torch.tensor(i) for i in x_featureemb1]
            x_featureemb2 = [torch.tensor(i) for i in x_featureemb2]
            x_featureemb3 = [torch.tensor(i) for i in x_featureemb3]
            x_featureemb4 = [torch.tensor(i) for i in x_featureemb4]
            x_featureemb5 = [torch.tensor(i) for i in x_featureemb5]
            x_feature = [torch.tensor(i) for i in x_feature]

            x_wordemb = nn.utils.rnn.pad_sequence(x_wordemb, batch_first=True, padding_value=0)
            x_featureemb0 = nn.utils.rnn.pad_sequence(x_featureemb0, batch_first=True, padding_value=0)
            x_featureemb1 = nn.utils.rnn.pad_sequence(x_featureemb1, batch_first=True, padding_value=0)
            x_featureemb2 = nn.utils.rnn.pad_sequence(x_featureemb2, batch_first=True, padding_value=0)
            x_featureemb3 = nn.utils.rnn.pad_sequence(x_featureemb3, batch_first=True, padding_value=0)
            x_featureemb4 = nn.utils.rnn.pad_sequence(x_featureemb4, batch_first=True, padding_value=0)
            x_featureemb5 = nn.utils.rnn.pad_sequence(x_featureemb5, batch_first=True, padding_value=0)
            x_feature = nn.utils.rnn.pad_sequence(x_feature, batch_first=True, padding_value=0)

            y = list(y[argsort_index])
            for i, val in enumerate(y):
                index = val[case]
                y[i] = np.zeros(x_wordemb.shape[1])
                y[i][index] = 1

            y = torch.tensor(y, dtype=torch.long)

            if gpu:
                x_wordemb = x_wordemb.cuda()
                x_featureemb0 = x_featureemb0.cuda()
                x_featureemb1 = x_featureemb1.cuda()
                x_featureemb2 = x_featureemb2.cuda()
                x_featureemb3 = x_featureemb3.cuda()
                x_featureemb4 = x_featureemb4.cuda()
                x_featureemb5 = x_featureemb5.cuda()
                x_feature = x_feature.cuda()
                y = y.cuda()

            out = bilstm.forward(x_wordemb, x_featureemb0, x_featureemb1, x_featureemb2, x_featureemb3, x_featureemb4, x_featureemb5, x_feature)
            
            out = torch.cat((out[:, :, 0].reshape(x_wordemb.shape[0], 1, -1), out[:, :, 1].reshape(x_wordemb.shape[0], 1, -1)), 1)
            pred = out.argmax(dim=2)[:, 1]
            correct += pred.eq(y.argmax(dim=1)).sum().item()
        print(f'[{epoch}] acc: {correct/N}')
        save_model(epoch, bilstm, gpu)


def save_model(epoch, bilstm, gpu):
    print('___save_model___')
    bilstm.cpu()
    torch.save(bilstm.state_dict(), f'./model/{epoch}.pkl')
    if gpu:
        bilstm.cuda()

def load_model(epoch, bilstm, domain):
    print('___load_model___')
    bilstm.load_state_dict(torch.load(f'./model_{domain}/{epoch}.pkl'))

def test(_tests, epoch, bilstm, model_domain, case='ga'):
    load_model(epoch, bilstm, model_domain)

    from tqdm import tqdm
    optimizer = optim.Adam(bilstm.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    correct_all = 0
    sum_length = 0
    for domain in domain_dict:
        tests = _tests[domain]
        running_loss = 0.0
        bilstm.eval()
        correct = 0
        for i in tqdm(range(0, len(tests))):
            batch = tests[i]

            x = [batch[0]]
            y = [batch[1]]
            argsort_index = np.array([i.shape[0] for i in x]).argsort()[::-1]
            x_wordemb = [np.array(i['単語ID'], dtype=np.int) for i in x]
            x_featureemb0 = [np.array(i['形態素0'], dtype=np.int) for i in x]
            x_featureemb1 = [np.array(i['形態素1'], dtype=np.int) for i in x]
            x_featureemb2 = [np.array(i['形態素2'], dtype=np.int) for i in x]
            x_featureemb3 = [np.array(i['形態素3'], dtype=np.int) for i in x]
            x_featureemb4 = [np.array(i['形態素4'], dtype=np.int) for i in x]
            x_featureemb5 = [np.array(i['形態素5'], dtype=np.int) for i in x]
            x_feature = [np.array(i[['n単語目', 'n文節目','is主辞', 'is_target_verb', '述語からの距離']], dtype=np.int32) for i in x]

            x_wordemb = np.array(x_wordemb)[argsort_index]
            x_featureemb0 = np.array(x_featureemb0)[argsort_index]
            x_featureemb1 = np.array(x_featureemb1)[argsort_index]
            x_featureemb2 = np.array(x_featureemb2)[argsort_index]
            x_featureemb3 = np.array(x_featureemb3)[argsort_index]
            x_featureemb4 = np.array(x_featureemb4)[argsort_index]
            x_featureemb5 = np.array(x_featureemb5)[argsort_index]
            x_feature = np.array(x_feature)[argsort_index]

            x_wordemb = [torch.tensor(i) for i in x_wordemb]
            x_featureemb0 = [torch.tensor(i) for i in x_featureemb0]
            x_featureemb1 = [torch.tensor(i) for i in x_featureemb1]
            x_featureemb2 = [torch.tensor(i) for i in x_featureemb2]
            x_featureemb3 = [torch.tensor(i) for i in x_featureemb3]
            x_featureemb4 = [torch.tensor(i) for i in x_featureemb4]
            x_featureemb5 = [torch.tensor(i) for i in x_featureemb5]
            x_feature = [torch.tensor(i) for i in x_feature]

            x_wordemb = nn.utils.rnn.pad_sequence(x_wordemb, batch_first=True, padding_value=0)
            x_featureemb0 = nn.utils.rnn.pad_sequence(x_featureemb0, batch_first=True, padding_value=0)
            x_featureemb1 = nn.utils.rnn.pad_sequence(x_featureemb1, batch_first=True, padding_value=0)
            x_featureemb2 = nn.utils.rnn.pad_sequence(x_featureemb2, batch_first=True, padding_value=0)
            x_featureemb3 = nn.utils.rnn.pad_sequence(x_featureemb3, batch_first=True, padding_value=0)
            x_featureemb4 = nn.utils.rnn.pad_sequence(x_featureemb4, batch_first=True, padding_value=0)
            x_featureemb5 = nn.utils.rnn.pad_sequence(x_featureemb5, batch_first=True, padding_value=0)
            x_feature = nn.utils.rnn.pad_sequence(x_feature, batch_first=True, padding_value=0)

            y = list(y)
            for i, val in enumerate(y):
                index = val[case]
                y[i] = np.zeros(x_wordemb.shape[1])
                y[i][index] = 1

            y = torch.tensor(y, dtype=torch.long)

            if gpu:
                x_wordemb = x_wordemb.cuda()
                x_featureemb0 = x_featureemb0.cuda()
                x_featureemb1 = x_featureemb1.cuda()
                x_featureemb2 = x_featureemb2.cuda()
                x_featureemb3 = x_featureemb3.cuda()
                x_featureemb4 = x_featureemb4.cuda()
                x_featureemb5 = x_featureemb5.cuda()
                x_feature = x_feature.cuda()
                y = y.cuda()

            out = bilstm.forward(x_wordemb, x_featureemb0, x_featureemb1, x_featureemb2, x_featureemb3, x_featureemb4, x_featureemb5, x_feature)
            
            out = torch.cat((out[:, :, 0].reshape(x_wordemb.shape[0], 1, -1), out[:, :, 1].reshape(x_wordemb.shape[0], 1, -1)), 1)
            pred = out.argmax(dim=2)[:, 1]
            correct += pred.eq(y.argmax(dim=1)).sum().item()
        print(f'[{domain}] acc: {correct/len(tests)}')
        correct_all += correct
        sum_length += len(tests)
    print(f'[All] acc: {correct_all/sum_length}')
    return correct_all/sum_length

test(_tests, 2, bilstm, 'Random')