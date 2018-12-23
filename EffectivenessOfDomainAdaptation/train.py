import argparse
from collections import defaultdict
import json
import numpy as np
import torch.optim as optim
import os
import pandas as pd
from pprint import pprint
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import re

import sys
sys.path.append('../utils')
from loader import WordVector, load_datasets, split, split_each_domain
from model import BiLSTM, FeatureAugmentation

def is_num(text):
    m = re.match('\A[0-9]+\Z', text)
    if m:
        return True
    else:
        return False

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
    emb_dim = 200
    class_num = 2
    is_gpu = True
    if gpu == -1:
        is_gpu = False
    if model=='Base' or model=='FT':
        bilstm = BiLSTM(emb_dim, class_num, vocab_size, v_vec, dropout_ratio, n_layers, is_gpu, )
    elif model == 'FA':
        bilstm = FeatureAugmentation(emb_dim, class_num, vocab_size, v_vec, dropout_ratio, n_layers, is_gpu)
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
                    batch = trains_dict[perm[i:i+batch_size]]
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
        _results, _ = test(vals, bilstm, batch_size, args)
        results[epoch] = _results
        if args.save:
            save_model(epoch, bilstm, args.dump_dir, args.gpu)
    if args.save:
        dump_dic(results, args.dump_dir, 'training_logs.json')
    best_epochs = defaultdict(lambda: defaultdict(float))
    for epoch in results:
        for domain in sorted(results[epoch].keys()):
            if results[epoch][domain]['F1']['F1-score']['total'] > best_epochs[domain]['F1-score(total)']:
                best_epochs[domain]['F1-score(total)'] = results[epoch][domain]['F1']['F1-score']['total']
                best_epochs[domain]['acc(one_label)'] = results[epoch][domain]['acc(one_label)']
                best_epochs[domain]['epoch'] = epoch
    if args.save:
        dump_dic(best_epochs, args.dump_dir, 'training_result.json')
    print('--- finish training ---\n--- best F1-score epoch for each domain ---')
    for domain in sorted(best_epochs.keys()):
        print(f'{domain} [epoch: {best_epochs[domain]["epoch"]}]\tF1-score: {best_epochs[domain]["F1-score(total)"]}\tacc(one_label): {best_epochs[domain]["acc(one_label)"]}')
    return 1 - best_epochs['All']["F1-score(total)"]

def initialize_confusion_matrix():
    case_types = ['none', 'exo1', 'exo2', 'exoX', 'intra(dep)', 'intra(dep)_false', 'intra(zero)', 'intra(zero)_false', 'inter(zero)', 'inter(zero)_false']
    index = pd.MultiIndex.from_arrays([['predicted']*10, case_types])
    case_types = ['none', 'exo1', 'exo2', 'exoX', 'intra(dep)', 'intra(zero)', 'inter(zero)']
    columns = pd.MultiIndex.from_arrays([['actual']*7, case_types])
    df = pd.DataFrame(data=0, index=index, columns=columns)
    return df

def calculate_confusion_matrix(confusion_matrix, _batch, _predict_index, target_case):
    if _predict_index == 0:
        predict_case_type = 'none'
    elif _predict_index == 1:
        predict_case_type = 'exoX'
    elif _predict_index == 2:
        predict_case_type = 'exo2'
    elif _predict_index == 3:
        predict_case_type = 'exo1'
    else:
        target_verb_index = _batch[1].name
        verb_phrase_number = _batch[0]['n文節目'][target_verb_index]
        if 'n文目' in _batch[0].keys() and _batch[0]['n文目'][target_verb_index] != _batch[0]['n文目'][_predict_index]:
            predict_case_type = 'inter(zero)'
        elif _predict_index >= len(_batch[0]):
            predict_case_type = 'inter(zero)'
        else:
            #文内解析時（文内解析時は，文間ゼロ照応に対して予測することはありえない）
            predict_dependency_relation_phrase_number = _batch[0]['係り先文節'][_predict_index]
            if verb_phrase_number == predict_dependency_relation_phrase_number:
                predict_case_type = 'intra(dep)'
            else:
                predict_case_type = 'intra(zero)'

    correct_case_index_list = [int(i) for i in _batch[1][target_case].split(',')]
    for correct_case_index in correct_case_index_list.copy():
        eq = _batch[0]['eq'][correct_case_index]
        if is_num(eq):
            correct_case_index_list += np.arange(len(_batch[0]))[_batch[0]['eq']==eq].tolist()
    if _predict_index in correct_case_index_list:
        #予測正解時
        actual_case_type = predict_case_type
        is_correct = True
    else:
        #予測不正解時
        actual_case_type = _batch[1][f'{target_case}_type'].split(',')[0]
        if predict_case_type == 'intra(dep)' or predict_case_type == 'intra(zero)' or predict_case_type == 'inter(zero)':
            predict_case_type += '_false'
        is_correct = False
    confusion_matrix['actual'][actual_case_type]['predicted'][predict_case_type] += 1
    return is_correct

def calculate_f1(confusion_matrix):
    case_types = ['none', 'exo1', 'exo2', 'exoX', 'intra(dep)', 'intra(zero)', 'inter(zero)']
    columns = ['precision', 'recall', 'F1-score']
    df = pd.DataFrame(data=0.0, index=case_types, columns=columns)
    for case_type in case_types:
        tp = confusion_matrix['actual', case_type]['predicted', case_type]
        #precision
        tp_fp = sum(confusion_matrix.loc['predicted', case_type])
        if case_type == 'intra(zero)' or case_type == 'intra(dep)' or case_type == 'inter(zero)':
            tp_fp += sum(confusion_matrix.loc['predicted', f'{case_type}_false'])
        if tp_fp != 0:
            df['precision'][case_type] = tp/tp_fp
        #recall
        if sum(confusion_matrix['actual', case_type]) != 0:
            df['recall'][case_type] = tp/sum(confusion_matrix['actual', case_type])
        #F1-score
        if (df['precision'][case_type]+df['recall'][case_type]) != 0:
            df['F1-score'][case_type] = (2*df['precision'][case_type]*df['recall'][case_type])/(df['precision'][case_type]+df['recall'][case_type])
    all_tp = 0
    all_tp_fp = 0
    all_tp_fn = 0
    for case_type in case_types:
        all_tp += confusion_matrix['actual', case_type]['predicted', case_type]
        all_tp_fp += sum(confusion_matrix.loc['predicted', case_type])
        all_tp_fn += sum(confusion_matrix['actual', case_type])
    df.loc['total'] = 0
    df['precision']['total'] = all_tp/(all_tp_fp)
    df['recall']['total'] = all_tp/(all_tp_fn)
    df['F1-score']['total'] = (2*df['precision']['total']*df['recall']['total'])/(df['precision']['total']+df['recall']['total'])
    return df

def predicted_log(batch, pred, target_case, corrects):
    batchsize = len(batch)
    for i in range(batchsize):
        target_verb_index = batch[i][1].name
        predicted_argument_index = pred[i].item()
        actual_argument_index = int(batch[i][1][target_case].split(',')[0])
        target_verb = batch[i][0]['単語'][target_verb_index]
        if predicted_argument_index >= len(batch[i][0]):
            predicted_argument = 'inter(zero)'
        else:
            predicted_argument = batch[i][0]['単語'][predicted_argument_index]

        actual_argument = batch[i][0]['単語'][actual_argument_index]
        sentence = ' '.join(batch[i][0]['単語'][4:])
        file = batch[i][2]
        log = {
            '正解': corrects[i],
            '述語位置': target_verb_index - 4,
            '述語': target_verb,
            '正解項位置': actual_argument_index - 4,
            '正解項': actual_argument,
            '予測項位置': predicted_argument_index - 4,
            '予測項': predicted_argument,
            '解析対象文': sentence,
            'ファイル': file
        }
        domain = return_file_domain(file)
        yield domain, log

def test(tests, bilstm, batch_size, args):
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
            N = len(tests[domain])
            perm = np.random.permutation(N)
            for i in range(0, N, batch_size):
                batch = tests[perm[i:i+batch_size]]
                batches.append(batch)
    else:
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
    emb_type = 'Word2VecWiki'
    wv = WordVector(emb_type, args.emb_path)
    is_intra = True
    datasets = load_datasets(wv, is_intra, args.media)
    trains_dict, vals_dict, tests_dict = split_each_domain(datasets)

    args.__dict__['trains_size'] = sum([len(trains_dict[domain]) for domain in args.media])
    args.__dict__['vals_size'] = sum([len(vals_dict[domain]) for domain in args.media])
    args.__dict__['tests_size'] = sum([len(tests_dict[domain]) for domain in args.media])

    bilstm = initialize_model(args.gpu, vocab_size=len(wv.index2word), v_vec= wv.vectors, dropout_ratio=0.2, n_layers=1, model=args.model)
    dump_dic(args.__dict__, args.dump_dir, 'args.json')
    pprint(args.__dict__)

    train(trains_dict, vals_dict, bilstm, args, lr=0.01, batch_size=16)

if __name__ == '__main__':
    main()
