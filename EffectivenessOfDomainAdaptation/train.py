import argparse
from collections import defaultdict
import numpy as np
import torch.optim as optim
import pandas as pd
from pprint import pprint
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from model import BiLSTM, FeatureAugmentation, ClassProbabilityShift

import sys
sys.path.append('../utils')
from loader import WordVector, load_datasets, split, split_each_domain
from store import dump_dict
from subfunc import return_file_domain, initialize_confusion_matrix, calculate_confusion_matrix, calculate_f1, predicted_log

# init model
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Embedding') == -1):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

def create_arg_parser():
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--epochs', '-e', dest='max_epoch', type=int, default=15, help='max epoch')
    parser.add_argument('--emb_path', dest='emb_path', help='word embedding path')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1, help='GPU ID for execution')
    parser.add_argument('--case', '-c', dest='case', type=str, required=True, choices=['ga', 'o', 'ni'], help='target "case" type')
    parser.add_argument('--media', '-m', dest='media', nargs='+', type=str, default=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], choices=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], help='training media type')
    parser.add_argument('--save', dest='save', action='store_true', default=False, help='saving model or not')
    parser.add_argument('--dump_dir', dest='dump_dir', type=str, required=True, help='model dump directory path')
    parser.add_argument('--model', dest='model', type=str, required=True, choices=['Base', 'FT', 'FA', 'CPS', 'VOT', 'MIX'])
    return parser

def initialize_model(gpu, vocab_size, v_vec, dropout_ratio, n_layers, model):
    is_gpu = True
    if gpu == -1:
        is_gpu = False
    if model=='Base' or model=='FT':
        bilstm = BiLSTM(vocab_size, v_vec, dropout_ratio, n_layers, gpu=is_gpu)
    elif model == 'FA':
        bilstm = FeatureAugmentation(vocab_size, v_vec, dropout_ratio, gpu=is_gpu)
    elif model == 'CPS':
        bilstm = ClassProbabilityShift(vocab_size, v_vec, dropout_ratio, statistics_of_each_case_type=None, gpu=is_gpu)
    if is_gpu:
        bilstm = bilstm.cuda()

    for m in bilstm.modules():
        print(m.__class__.__name__)
        weights_init(m)

    return bilstm

def translate_df_tensor(df_list, keys, gpu_id):
    vec = [np.array(i[keys], dtype=np.int) for i in df_list]
    vec = np.array(vec)
    vec = [torch.tensor(i) for i in vec]
    vec = nn.utils.rnn.pad_sequence(vec, batch_first=True, padding_value=0)
    if gpu_id >= 0:
        vec = vec.cuda()
    return vec

def translate_df_y(df_list, keys, gpu_id):
    vec = [int(i[keys].split(',')[0]) for i in df_list]
    vec = torch.tensor(vec)
    if gpu_id >= 0:
        vec = vec.cuda()
    return vec

def translate_batch(batch, gpu, case):
    x = batch[:, 0]
    y = batch[:, 1]
    files = batch[:, 2]
    batchsize = len(batch)

    max_length = x[0].shape[0]
    sentences = [i['単語'].values[4:] for i in batch[:, 0]]
    sentences = np.array(sentences)
    x_wordID = translate_df_tensor(x, ['単語ID'], gpu)
    x_wordID = x_wordID.reshape(batchsize, -1)
    x_feature_emb_list = []
    for i in range(6):
        x_feature_emb = translate_df_tensor(x, [f'形態素{i}'], gpu)
        x_feature_emb = x_feature_emb.reshape(batchsize, -1)
        x_feature_emb_list.append(x_feature_emb)

    x_feature = translate_df_tensor(x, ['n文節目','is主辞', 'is_target_verb', '述語からの距離'], gpu)
    x = [x_wordID, x_feature_emb_list, x_feature]

    y = translate_df_y(y, case, -1)
    y = y.reshape(batchsize)
    y = torch.eye(max_length, dtype=torch.long)[y]
    if gpu >= 0:
        y = y.cuda()

    return x, y, files

def train(trains_dict, vals_dict, bilstm, args, lr, batch_size):
    print('--- start training ---')
    epochs = args.max_epoch+1
    results = {}
    optimizer = optim.Adam(bilstm.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs):
        batches = []
        if args.model =='FA' or args.model == 'MIX':
            for domain in args.media:
                N = len(trains_dict[domain])
                perm = np.random.permutation(N)
                for i in range(0, N, batch_size):
                    batch = trains_dict[domain][perm[i:i+batch_size]]
                    batches.append(batch)
        else:
            trains = np.hstack(trains_dict.values())
            N = len(trains)
            perm = np.random.permutation(N)
            for i in range(0, N, batch_size):
                batch = trains[perm[i:i+batch_size]]
                batches.append(batch)
        random.shuffle(batches)
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        bilstm.train()
        for batch in tqdm(batches):
            bilstm.zero_grad()
            optimizer.zero_grad()
            #0 paddingするために，長さで降順にソートする．
            argsort_index = np.array([i.shape[0] for i in batch[:, 0]]).argsort()[::-1]
            batch = batch[argsort_index]
            x, y, files = translate_batch(batch, args.gpu, args.case)
            batchsize = len(batch)
            if args.model == 'FA' or args.model == 'MIX':
                domain = return_file_domain(files[0])
                out = bilstm.forward(x, domain)
            else:
                out = bilstm.forward(x)
            out = torch.cat((out[:, :, 0].reshape(batchsize, 1, -1), out[:, :, 1].reshape(batchsize, 1, -1)), dim=1)
            pred = out.argmax(dim=2)[:, 1]
            running_correct += pred.eq(y.argmax(dim=1)).sum().item()
            running_samples += len(batch)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'[epoch: {epoch}]\tloss: {running_loss/(running_samples/batch_size)}\tacc(one_label): {running_correct/running_samples}')
        _results, _ = test(vals_dict, bilstm, batch_size, args)
        results[epoch] = _results
        if args.save:
            save_model(epoch, bilstm, args.dump_dir, args.gpu)
    if args.save:
        dump_dict(results, args.dump_dir, 'training_logs')
    best_epochs = defaultdict(lambda: defaultdict(float))
    for epoch in results:
        for domain in sorted(results[epoch].keys()):
            if results[epoch][domain]['F1']['F1-score']['total'] > best_epochs[domain]['F1-score(total)']:
                best_epochs[domain]['F1-score(total)'] = results[epoch][domain]['F1']['F1-score']['total']
                best_epochs[domain]['acc(one_label)'] = results[epoch][domain]['acc(one_label)']
                best_epochs[domain]['epoch'] = epoch
    if args.save:
        dump_dict(best_epochs, args.dump_dir, 'training_result')
    print('--- finish training ---\n--- best F1-score epoch for each domain ---')
    for domain in sorted(best_epochs.keys()):
        print(f'{domain} [epoch: {best_epochs[domain]["epoch"]}]\tF1-score: {best_epochs[domain]["F1-score(total)"]}\tacc(one_label): {best_epochs[domain]["acc(one_label)"]}')
    return 1 - best_epochs['All']["F1-score(total)"]


def test(tests_dict, bilstm, batch_size, args):
    results = defaultdict(lambda: defaultdict(float))
    logs = defaultdict(list)
    for domain in args.media:
        results[domain]['confusion_matrix'] = initialize_confusion_matrix()
    results['All']['confusion_matrix'] = initialize_confusion_matrix()
    bilstm.eval()
    criterion = nn.CrossEntropyLoss()
    batches = []
    if args.model =='FA' or args.model == 'MIX':
        for domain in args.media:
            N = len(tests_dict[domain])
            perm = np.random.permutation(N)
            for i in range(0, N, batch_size):
                batch = tests_dict[domain][perm[i:i+batch_size]]
                batches.append(batch)
    else:
        tests = np.hstack(tests_dict.values())
        N = len(tests)
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            batch = tests[perm[i:i+batch_size]]
            batches.append(batch)
    random.shuffle(batches)
    for batch in tqdm(batches):
        batchsize = len(batch)

        #0 paddingするために，長さで降順にソートする．
        argsort_index = np.array([i.shape[0] for i in batch[:, 0]]).argsort()[::-1]
        batch = batch[argsort_index]
        x, y, files = translate_batch(batch, args.gpu, args.case)

        if args.model == 'FA' or args.model == 'MIX':
            domain = return_file_domain(files[0])
            out = bilstm.forward(x, domain)
        else:
            out = bilstm.forward(x)
        out = torch.cat((out[:, :, 0].reshape(batchsize, 1, -1), out[:, :, 1].reshape(batchsize, 1, -1)), dim=1)

        pred = out.argmax(dim=2)[:, 1]
        corrects = []
        for j, file in enumerate(files):
            correct = pred[j].eq(y[j].argmax()).item()
            domain = return_file_domain(file)
            results[domain]['correct'] += correct
            results[domain]['samples'] += 1
            loss = criterion(out[j].reshape(1, 2, -1), y[j].reshape(1, -1))
            results[domain]['loss'] += loss.item()
            correct = calculate_confusion_matrix(results[domain]['confusion_matrix'], batch[j], pred[j].item(), args.case)
            corrects.append(correct)
        for domain, log in predicted_log(batch, pred, args.case, corrects):
            logs[domain].append(log)

    for domain in args.media:
        results['All']['loss'] += results[domain]['loss']
        results['All']['samples'] += results[domain]['samples']
        results['All']['correct'] += results[domain]['correct']
        for i in range(results[domain]['confusion_matrix'].shape[0]):
            for j in range(results[domain]['confusion_matrix'].shape[1]):
                results['All']['confusion_matrix'].iat[i, j] += results[domain]['confusion_matrix'].iat[i, j]
        results[domain]['loss'] /= results[domain]['samples']
        results[domain]['acc(one_label)'] = results[domain]['correct']/results[domain]['samples']
        results[domain]['F1'] = calculate_f1(results[domain]['confusion_matrix'])
    results['All']['loss'] /= results['All']['samples']
    results['All']['acc(one_label)'] = results['All']['correct']/results['All']['samples']
    results['All']['F1'] = calculate_f1(results['All']['confusion_matrix'])
    for domain in sorted(results.keys()):
        print(f'[domain: {domain}]\ttest loss: {results[domain]["loss"]}\tF1-score: {results[domain]["F1"]["F1-score"]["total"]}\tacc(one_label): {results[domain]["acc(one_label)"]}')
        results[domain]['confusion_matrix'] = results[domain]['confusion_matrix'].to_dict()
        tmp_dict1 = {}
        for key1 in results[domain]['confusion_matrix']:
            tmp_dict2 = {}
            for key2 in results[domain]['confusion_matrix'][key1]:
                tmp_dict2['_'.join(key2)] = results[domain]['confusion_matrix'][key1][key2]
            tmp_dict1['_'.join(key1)] = tmp_dict2
        results[domain]['confusion_matrix'] = tmp_dict1
        results[domain]['F1'] = results[domain]['F1'].to_dict()
    return results, logs

def init_statistics_of_each_case_type(trains_y, case_type):
    case_type_counts = defaultdict(int)
    for case_types in trains_y[f'{case}_type']:
        case_type = case_types.split(',')[0]
        case_type_counts[case_type] += 1
    return case_type_counts

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    emb_type = 'Word2VecWiki'
    wv = WordVector(emb_type, args.emb_path)
    is_intra = True
    datasets = load_datasets(wv, is_intra, args.media)
    trains_dict, vals_dict, tests_dict = split_each_domain(datasets)
    import ipdb; ipdb.set_trace();

    args.__dict__['trains_size'] = sum([len(trains_dict[domain]) for domain in args.media])
    args.__dict__['vals_size'] = sum([len(vals_dict[domain]) for domain in args.media])
    args.__dict__['tests_size'] = sum([len(tests_dict[domain]) for domain in args.media])

    bilstm = initialize_model(args.gpu, vocab_size=len(wv.index2word), v_vec= wv.vectors, dropout_ratio=0.2, n_layers=1, model=args.model)
    dump_dict(args.__dict__, args.dump_dir, 'args')
    pprint(args.__dict__)

    train(trains_dict, vals_dict, bilstm, args, lr=0.01, batch_size=16)

if __name__ == '__main__':
    main()
