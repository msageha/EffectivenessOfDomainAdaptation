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
import json

import train

import sys
sys.path.append('../utils')
from loader import DatasetLoading, load_model
from store import dump_dict, dump_predict_logs
from subfunc import return_file_domain, predicted_log
from calc_result import ConfusionMatrix


def run(tests_dict, bilstm_FT, bilstm_FA, bilstm_CPS, batch_size, args):
    results = defaultdict(lambda: defaultdict(float))
    logs = defaultdict(list)
    for domain in tests_dict.keys():
        results[domain]['confusion_matrix'] = ConfusionMatrix()
    results['All']['confusion_matrix'] = ConfusionMatrix()

    bilstm_FT.eval()
    bilstm_FA.eval()
    bilstm_CPS.eval()
    criterion = nn.CrossEntropyLoss()
    batches = []
    for domain in tests_dict.keys():
        N = len(tests_dict[domain])
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            batch = tests_dict[domain][perm[i:i+batch_size]]
            batches.append(batch)
    random.shuffle(batches)
    for batch in tqdm(batches):
        batchsize = len(batch)

        #0 paddingするために，長さで降順にソートする．
        argsort_index = np.array([i.shape[0] for i in batch[:, 0]]).argsort()[::-1]
        batch = batch[argsort_index]
        x, y, files = train.translate_batch(batch, args.gpu, args.case)
        # fine tuning
        out_FT = bilstm_FT.forward(x)
        out_FT = torch.cat((out_FT[:, :, 0].reshape(batchsize, 1, -1), out_FT[:, :, 1].reshape(batchsize, 1, -1)), dim=1)
        pred_FT = out_FT.argmax(dim=2)[:, 1]

        # feature augumentation
        domain = return_file_domain(files[0])
        out_FA = bilstm_FA.forward(x, domain)
        out_FA = torch.cat((out_FA[:, :, 0].reshape(batchsize, 1, -1), out_FA[:, :, 1].reshape(batchsize, 1, -1)), dim=1)
        pred_FA = out_FA.argmax(dim=2)[:, 1]

        # class probability shift
        domains = [return_file_domain(file) for file in files]
        out_CPS = bilstm_CPS.forward(x, domains)
        out_CPS = torch.cat((out_CPS[:, :, 0].reshape(batchsize, 1, -1), out_CPS[:, :, 1].reshape(batchsize, 1, -1)), dim=1)
        pred_CPS = out_CPS.argmax(dim=2)[:, 1]

        pred = []
        pred_same_num = []
        pred_same2_FT_adopt = 0
        pred_same2_FA_adopt = 0
        pred_same2_CPS_adopt = 0
        pred_same1_FT_adopt = 0
        pred_same1_FA_adopt = 0
        pred_same1_CPS_adopt = 0
        for i in range(batchsize):
            # 3モデルともに，同じ予測
            if pred_FT[i].item() == pred_FA[i].item() and pred_FT[i].item() == pred_CPS[i].item():
                _pred = pred_FT[i]
                _num = 3
            # 2モデルともに，同じ予測
            elif pred_FT[i].item() == pred_FA[i].item():
                # CPSだけ間違えた
                _pred = pred_FT[i]
                _num = 2
                pred_same2_FT_adopt += 1
                pred_same2_FA_adopt += 1
            elif pred_FA[i].item() == pred_CPS[i].item():
                # FTだけ間違えた
                _pred = pred_FA[i]
                _num = 2
                pred_same2_FA_adopt += 1
                pred_same2_CPS_adopt += 1
            elif pred_FT[i].item() == pred_CPS[i].item():
                # FAだけ間違えた
                _pred = pred_CPS[i]
                _num = 2
                pred_same2_FT_adopt += 1
                pred_same2_CPS_adopt += 1
            # すべて違う予測
            else:
                _num = 1
                if out_FT[i][1][pred_FT[i]] == max([out_FT[i][1][pred_FT[i]], out_FA[i][1][pred_FA[i]], out_CPS[i][1][pred_CPS[i]]]):
                    _pred = pred_FT[i]
                    pred_same1_FT_adopt += 1
                elif out_FA[i][1][pred_FA[i]] == max([out_FT[i][1][pred_FT[i]], out_FA[i][1][pred_FA[i]], out_CPS[i][1][pred_CPS[i]]]):
                    _pred = pred_FA[i]
                    pred_same1_FA_adopt += 1
                elif out_CPS[i][1][pred_CPS[i]] == max([out_FT[i][1][pred_FT[i]], out_FA[i][1][pred_FA[i]], out_CPS[i][1][pred_CPS[i]]]):
                    _pred = pred_CPS[i]
                    pred_same1_CPS_adopt += 1
            pred.append(_pred)
            pred_same_num.append(_num)

        corrects = []
        for j, file in enumerate(files):
            correct = pred[j].eq(y[j].argmax()).item()
            domain = return_file_domain(file)
            results[domain]['correct'] += correct
            if pred_same_num[j] == 3:
                results[domain]['correct_by_same3'] += correct
            elif pred_same_num[j] == 2:
                results[domain]['correct_by_same2'] += correct
            elif pred_same_num[j] == 1:
                results[domain]['correct_by_same1'] += correct
            results[domain]['samples'] += 1
            results[domain]['pred_same3'] += pred_same_num.count(3)
            results[domain]['pred_same2'] += pred_same_num.count(2)
            results[domain]['pred_same1'] += pred_same_num.count(1)
            results[domain]['pred_same2_FT_adopt'] += pred_same2_FT_adopt
            results[domain]['pred_same2_FA_adopt'] += pred_same2_FA_adopt
            results[domain]['pred_same2_CPS_adopt'] += pred_same2_CPS_adopt
            results[domain]['pred_same1_FT_adopt'] += pred_same1_FT_adopt
            results[domain]['pred_same1_FA_adopt'] += pred_same1_FA_adopt
            results[domain]['pred_same1_CPS_adopt'] += pred_same1_CPS_adopt
            correct = results[domain]['confusion_matrix'].calculate(batch[j], pred[j].item(), args.case)
            corrects.append(correct)
        for domain, log in predicted_log(batch, pred, args.case, corrects):
            logs[domain].append(log)

    for domain in tests_dict.keys():
        results['All']['loss'] += results[domain]['loss']
        results['All']['samples'] += results[domain]['samples']
        results['All']['correct'] += results[domain]['correct']
        for i in range(results[domain]['confusion_matrix'].df.shape[0]):
            for j in range(results[domain]['confusion_matrix'].df.shape[1]):
                results['All']['confusion_matrix'].df.iat[i, j] += results[domain]['confusion_matrix'].df.iat[i, j]
        results[domain]['loss'] /= results[domain]['samples']
        results[domain]['acc(one_label)'] = results[domain]['correct']/results[domain]['samples']
        results[domain]['F1'] = results[domain]['confusion_matrix'].calculate_f1()
    results['All']['loss'] /= results['All']['samples']
    results['All']['acc(one_label)'] = results['All']['correct']/results['All']['samples']
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
    parser.add_argument('--load_FTdir', dest='load_FTdir', type=str, required=True, help='fine tuning model load directory path')
    parser.add_argument('--load_FAdir', dest='load_FAdir', type=str, required=True, help='feature augumentation model load directory path')
    parser.add_argument('--load_CPSdir', dest='load_CPSdir', type=str, required=True, help='class probability shift model load directory path')
    parser.add_argument('--dump_dir', dest='dump_dir', type=str, required=True, help='dump directory path')
    return parser


def max_f1_epochs_of_vals(train_result_path):
    with open(f'{train_result_path}/training_result.json') as f:
        val_results = json.load(f)
    return val_results


def load_config(args, directory):
    with open(f'{directory}/args.json') as f:
        params = json.load(f)
    for key in params:
        if key in args.__dict__:
            continue
        args.__dict__[key] = params[key]

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    load_config(args, args.load_FAdir)
    load_config(args, args.load_CPSdir)

    emb_type = 'Word2VecWiki'

    dl = DatasetLoading(emb_type, args.emb_path, exo1_word='僕', exo2_word='おまえ', exoX_word='これ')
    dl.making_intra_df()

    trains_dict, _, tests_dict = dl.split_each_domain('intra')

    statistics_of_each_case_type = train.init_statistics_of_each_case_type(trains_dict, args.case, args.media)

    bilstm_FT = train.initialize_model(args.gpu, vocab_size=len(dl.wv.index2word), v_vec=dl.wv.vectors, dropout_ratio=0.2, n_layers=3, model='Base', statistics_of_each_case_type=None)
    bilstm_FA = train.initialize_model(args.gpu, vocab_size=len(dl.wv.index2word), v_vec=dl.wv.vectors, dropout_ratio=0.2, n_layers=3, model='FA', statistics_of_each_case_type=None)
    bilstm_CPS = train.initialize_model(args.gpu, vocab_size=len(dl.wv.index2word), v_vec=dl.wv.vectors, dropout_ratio=0.2, n_layers=3, model='CPS', statistics_of_each_case_type=statistics_of_each_case_type)

    results = {}
    logs = {}
    # domain = 'All'

    pprint(args.__dict__)
    for domain in args.media:
        load_config(args, args.load_FTdir+f'/{domain}/{args.case}')
        val_results_FT = max_f1_epochs_of_vals(args.load_FTdir+f'/{domain}/{args.case}')
        epoch_FT = val_results_FT[domain]['epoch']
        val_results_FA = max_f1_epochs_of_vals(args.load_FAdir)
        epoch_FA = val_results_FA[domain]['epoch']
        val_results_CPS = max_f1_epochs_of_vals(args.load_CPSdir)
        epoch_CPS = val_results_CPS[domain]['epoch']

        load_model(epoch_FT, bilstm_FT, args.load_FTdir+f'/{domain}/{args.case}', args.gpu)
        load_model(epoch_FA, bilstm_FA, args.load_FAdir, args.gpu)
        load_model(epoch_CPS, bilstm_CPS, args.load_CPSdir, args.gpu)

        _results, _logs = run(tests_dict, bilstm_FT, bilstm_FA, bilstm_CPS, 1, args)
        results[domain] = _results[domain]
        results[domain]['epoch_FT'] = epoch_FT
        results[domain]['epoch_FA'] = epoch_FA
        results[domain]['epoch_CPS'] = epoch_CPS
        logs[domain] = _logs[domain]
        dump_dict(results, args.dump_dir+f'/{domain}/{args.case}', 'test_logs')
        dump_predict_logs(logs, args.dump_dir+f'/{domain}/{args.case}')


if __name__ == '__main__':
    main()