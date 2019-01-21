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

from model import BiLSTM, OneHot, FeatureAugmentation, ClassProbabilityShift, Mixture
import test

import sys
sys.path.append('../utils')
from loader import DatasetLoading, load_model
from store import dump_dict, save_model
from subfunc import return_file_domain, predicted_log
from calc_result import ConfusionMatrix


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
    parser.add_argument('--media', '-m', dest='media', nargs='+', type=str, default=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], choices=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], help='training media type')
    parser.add_argument('--save', dest='save', action='store_true', default=False, help='saving model or not')
    parser.add_argument('--dump_dir', dest='dump_dir', type=str, required=True, help='model dump directory path')
    return parser


def initialize_model(gpu, vocab_size, v_vec, dropout_ratio, n_layers):
    is_gpu = True
    if gpu == -1:
        is_gpu = False
    bilstm = BiLSTM(vocab_size, v_vec, dropout_ratio, n_layers, gpu=is_gpu)
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


def translate_batch(batch, gpu):
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

    x_feature = translate_df_tensor(x, ['n文節目','is主辞'], gpu)
    x = [x_wordID, x_feature_emb_list, x_feature]

    y = translate_df_tensor(y, 'is_verb', gpu)

    return x, y, files


def run(trains, vals, bilstm, args, lr, batch_size):
    print('--- start training ---')
    epochs = args.max_epoch+1
    results = {}
    optimizer = optim.Adam(bilstm.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs):
        batches = []
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
            x, y, files = translate_batch(batch, args.gpu)
            batchsize = len(batch)
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
        _results, _ = test.run(vals_dict, bilstm, batch_size, args)
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


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    emb_type = 'Word2VecWiki'
    emb_path = '../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt'

    dl = DatasetLoading(emb_type, emb_path)
    dl.making_predicate_df()

    trains, vals, tests = dl.split(args.dataset_type)
    args.__dict__['trains_size'] = len(trains)
    args.__dict__['vals_size'] = len(vals)
    args.__dict__['tests_size'] = len(tests)

    bilstm = initialize_model(args.gpu, vocab_size=len(dl.wv.index2word), v_vec=dl.wv.vectors, dropout_ratio=0.2, n_layers=3)
    dump_dict(args.__dict__, args.dump_dir, 'args')
    pprint(args.__dict__)

    run(trains, vals, bilstm, args, lr=0.001, batch_size=64)

if __name__ == '__main__':
    main()
