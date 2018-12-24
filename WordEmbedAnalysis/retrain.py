import argparse
from allennlp.modules import elmo
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
import re

from model import BiLSTM
import train
# from train import translate_batch, initialize_model
import test

import sys
sys.path.append('../utils')
from loader import DatasetLoading, load_model, load_config
from store import save_model, dump_dict


def create_arg_parser():
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--epochs', '-e', dest='max_epoch', type=int, default=15, help='max epoch')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1, help='GPU ID for execution')
    parser.add_argument('--load_dir', dest='load_dir', type=str, required=True, help='model load directory path')
    return parser


def return_even_epochs(path):
    files = os.listdir(f'./{path}/model/')
    files = [int(file.split('.')[0]) for file in files]
    even_epochs = sorted(files)
    return even_epochs


def calc_even_results(vals, bilstm, args):
    results = {}
    even_epochs = return_even_epochs(args.dump_dir)
    for epoch in even_epochs:
        load_model(epoch, bilstm, args.load_dir, args.gpu)
        _results, _ = test.run(vals, bilstm, args)
        results[epoch] = _results
    return results, epoch


def run(trains, vals, bilstm, args):
    print('--- start retraining ---')
    epochs = args.max_epoch+1
    lr = 0.001  # 学習係数
    results, latest_epoch = calc_even_results(vals, bilstm, args)
    optimizer = optim.Adam(bilstm.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(latest_epoch+1, epochs):
        N = len(trains)
        perm = np.random.permutation(N)
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        bilstm.train()
        for i in tqdm(range(0, N, args.batch_size)):
            bilstm.zero_grad()
            optimizer.zero_grad()
            batch = trains[perm[i:i+args.batch_size]]
            # 0 paddingするために，長さで降順にソートする．
            argsort_index = np.array([i.shape[0] for i in batch[:, 0]]).argsort()[::-1]
            batch = batch[argsort_index]
            x, y, _ = train.translate_batch(batch, args.gpu, args.case, args.emb_type)
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

        print(f'[epoch: {epoch}]\tloss: {running_loss/(running_samples/args.batch_size)}\tacc(one_label): {running_correct/running_samples}')
        _results, _ = test(vals, bilstm, args)
        results[epoch] = _results
        save_model(epoch, bilstm, args.dump_dir, args.gpu)
    dump_dict(results, args.dump_dir, 'training_logs')
    best_epochs = defaultdict(lambda: defaultdict(float))
    for epoch in results:
        for domain in sorted(results[epoch].keys()):
            if results[epoch][domain]['F1']['F1-score']['total'] > best_epochs[domain]['F1-score(total)']:
                best_epochs[domain]['F1-score(total)'] = results[epoch][domain]['F1']['F1-score']['total']
                best_epochs[domain]['acc(one_label)'] = results[epoch][domain]['acc(one_label)']
                best_epochs[domain]['epoch'] = epoch
    dump_dict(best_epochs, args.dump_dir, 'training_result')
    print('--- finish training ---\n--- best F1-score epoch for each domain ---')
    for domain in sorted(best_epochs.keys()):
        print(f'{domain} [epoch: {best_epochs[domain]["epoch"]}]\tF1-score: {best_epochs[domain]["F1-score(total)"]}\tacc(one_label): {best_epochs[domain]["acc(one_label)"]}')


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    load_config(args)

    dl = DatasetLoading(args.emb_type, args.emb_path, media=args.media)
    if args.dataset_type == 'intra':
        dl.making_intra_df()
    elif args.dataset_type == 'inter':
        dl.making_inter_df()
    else:
        raise ValueError()

    trains, vals, _ = dl.split(args.dataset_type)
    bilstm = train.initialize_model(args.gpu, vocab_size=len(dl.wv.index2word), v_vec=dl.wv.vectors, emb_requires_grad=args.emb_requires_grad, args=args)

    pprint(args.__dict__)

    run(trains, vals, bilstm, args)


if __name__ == '__main__':
    main()
