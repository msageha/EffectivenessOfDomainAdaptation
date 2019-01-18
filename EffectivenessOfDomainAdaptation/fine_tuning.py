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
import train
import test

import sys
sys.path.append('../utils')
from loader import DatasetLoading, load_model, load_config
from store import dump_dict, save_model
from subfunc import return_file_domain, predicted_log
from calc_result import ConfusionMatrix


def create_arg_parser():
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--epochs', '-e', dest='max_epoch', type=int, default=15, help='max epoch')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1, help='GPU ID for execution')
    parser.add_argument('--load_dir', dest='load_dir', type=str, required=True, help='model load directory path')
    parser.add_argument('--model', dest='model', type=str, required=True, choices=['Base', 'MIX'])
    parser.add_argument('--dump_dir', dest='dump_dir')
    return parser


def run(trains, vals_dict, bilstm, args, ft_domain, lr, batch_size):
    print('--- start fine_tuning ---')
    epochs = args.max_epoch + 1
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
            x, y, files = train.translate_batch(batch, args.gpu, args.case)
            batchsize = len(batch)
            if args.model == 'MIX':
                domains = [return_file_domain(file) for file in files]
                out = bilstm.forward(x, domains)
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
        _results, _ = test.run(vals_dict, bilstm, batch_size, args)
        results[epoch] = _results
        if args.save:
            save_model(epoch, bilstm, args.dump_dir+f'/{ft_domain}/{args.case}', args.gpu)
    if args.save:
        dump_dict(results, args.dump_dir+f'/{ft_domain}/{args.case}', 'training_logs')
    best_epochs = defaultdict(lambda: defaultdict(float))
    for epoch in results:
        for domain in sorted(results[epoch].keys()):
            if results[epoch][domain]['F1']['F1-score']['total'] > best_epochs[domain]['F1-score(total)']:
                best_epochs[domain]['F1-score(total)'] = results[epoch][domain]['F1']['F1-score']['total']
                best_epochs[domain]['acc(one_label)'] = results[epoch][domain]['acc(one_label)']
                best_epochs[domain]['epoch'] = epoch
    if args.save:
        dump_dict(best_epochs, args.dump_dir+f'/{ft_domain}/{args.case}', 'training_result')
    print('--- finish training ---\n--- best F1-score epoch for each domain ---')
    for domain in sorted(best_epochs.keys()):
        print(f'{domain} [epoch: {best_epochs[domain]["epoch"]}]\tF1-score: {best_epochs[domain]["F1-score(total)"]}\tacc(one_label): {best_epochs[domain]["acc(one_label)"]}')


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    load_config(args)

    emb_type = 'Word2VecWiki'

    # dl = DatasetLoading(emb_type, args.emb_path, exo1_word='僕', exo2_word='おまえ', exoX_word='これ')
    # dl.making_intra_df()

    # trains_dict, vals_dict, _ = dl.split_each_domain('intra')

    # if args.model == 'MIX':
    #     statistics_of_each_case_type = train.init_statistics_of_each_case_type(trains_dict, args.case, args.media)
    # else:
    #     statistics_of_each_case_type = None

    # bilstm = train.initialize_model(args.gpu, vocab_size=len(dl.wv.index2word), v_vec=dl.wv.vectors, dropout_ratio=0.2, n_layers=3, model=args.model, statistics_of_each_case_type=statistics_of_each_case_type)

    pprint(args.__dict__)
    val_results = test.max_f1_epochs_of_vals(args.load_dir)

    for domain in ['OC', 'OY', 'OW', 'PB', 'PM', 'PN']:
        print(f'--- start {domain} fine tuning ---')
        dump_dict(args.__dict__, args.dump_dir+f'/{ft_domain}/{args.case}', 'args')
        # epoch = val_results[domain]['epoch']
        # load_model(epoch, bilstm, args.load_dir, args.gpu)
        
        # #lr = 0.0001にしてもいいかも
        # run(trains_dict[domain], vals_dict, bilstm, args, ft_domain=domain, lr=0.0001, batch_size=64)


if __name__ == '__main__':
    main()