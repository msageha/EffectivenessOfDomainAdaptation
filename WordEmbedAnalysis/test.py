import argparse
import json
from pprint import pprint
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import train

import sys
sys.path.append('../utils')

from loader import DatasetLoading, load_config, load_model
from calc_result import ConfusionMatrix
from store import dump_dict, dump_predict_logs
from subfunc import return_file_domain, predicted_log


def run(tests, bilstm, args):
    results = defaultdict(lambda: defaultdict(float))
    logs = defaultdict(list)
    for domain in args.media:
        results[domain]['confusion_matrix'] = ConfusionMatrix()
    results['All']['confusion_matrix'] = ConfusionMatrix()

    bilstm.eval()
    criterion = nn.CrossEntropyLoss()
    N = len(tests)
    for i in tqdm(range(0, N, args.batch_size)):
        batch = tests[i:i + args.batch_size]
        batchsize = len(batch)

        # 0 paddingするために，長さで降順にソートする．
        argsort_index = np.array([i.shape[0] for i in batch[:, 0]]).argsort()[::-1]
        batch = batch[argsort_index]
        x, y, files = train.translate_batch(batch, args.gpu, args.case, args.emb_type)

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
            correct = results[domain]['confusion_matrix'].calculate(batch[j], pred[j].item(), args.case)
            corrects.append(correct)
        for domain, log in predicted_log(batch, pred, args.case, corrects):
            logs[domain].append(log)

    for domain in args.media:
        results['All']['loss'] += results[domain]['loss']
        results['All']['samples'] += results[domain]['samples']
        results['All']['correct'] += results[domain]['correct']
        for i in range(results[domain]['confusion_matrix'].df.shape[0]):
            for j in range(results[domain]['confusion_matrix'].df.shape[1]):
                results['All']['confusion_matrix'].df.iat[i, j] += results[domain]['confusion_matrix'].df.iat[i, j]
        results[domain]['loss'] /= results[domain]['samples']
        results[domain]['acc(one_label)'] = results[domain]['correct'] / results[domain]['samples']
        results[domain]['F1'] = results[domain]['confusion_matrix'].calculate_f1()
    results['All']['loss'] /= results['All']['samples']
    results['All']['acc(one_label)'] = results['All']['correct'] / results['All']['samples']
    results['All']['F1'] = results['All']['confusion_matrix'].calculate_f1()
    for domain in sorted(results.keys()):
        print(f'[domain: {domain}]\ttest loss: {results[domain]["loss"]}\tF1-score: {results[domain]["F1"]["F1-score"]["total"]}\tacc(one_label): {results[domain]["acc(one_label)"]}')
        results[domain]['confusion_matrix'] = results[domain]['confusion_matrix'].df.to_dict()
        tmp_dict1 = {}
        for key1 in results[domain]['confusion_matrix']:
            tmp_dict2 = {}
            for key2 in results[domain]['confusion_matrix'][key1]:
                tmp_dict2['_'.join(key2)] = results[domain]['confusion_matrix'][key1][key2]
            tmp_dict1['_'.join(key1)] = tmp_dict2
        results[domain]['confusion_matrix'] = tmp_dict1
        results[domain]['F1'] = results[domain]['F1'].to_dict()
    return results, logs


def create_arg_parser():
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1, help='GPU ID for execution')
    parser.add_argument('--load_dir', dest='load_dir', type=str, required=True, help='model load directory path')
    return parser


def max_f1_epochs_of_vals(train_result_path):
    with open(f'{train_result_path}/training_result.json') as f:
        val_results = json.load(f)
    return val_results


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

    _, _, tests = dl.split(args.dataset_type)

    bilstm = train.initialize_model(args.gpu, vocab_size=len(dl.wv.index2word), v_vec= dl.wv.vectors, emb_requires_grad=args.emb_requires_grad, args=args)

    pprint(args.__dict__)
    val_results = max_f1_epochs_of_vals(args.load_dir)
    results = {}
    logs = {}
    domain = 'All'
    epoch = val_results[domain]['epoch']
    load_model(epoch, bilstm, args.load_dir, args.gpu)
    _results, _ = run(tests, bilstm, args)
    results[domain] = _results[domain]
    results[domain]['epoch'] = epoch
    for domain in args.media:
        epoch = val_results[domain]['epoch']
        load_model(epoch, bilstm, args.load_dir, args.gpu)
        _results, _logs = run(tests, bilstm, args)
        results[domain] = _results[domain]
        results[domain]['epoch'] = epoch
        logs[domain] = _logs[domain]
    dump_dict(results, args.load_dir, 'test_logs')
    dump_predict_logs(logs, args.load_dir)

if __name__ == '__main__':
    main()
