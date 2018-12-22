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

import sys
sys.path.append('../utils')
from loader import WordVector, load_datasets, split
from model import BiLSTM

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
    parser.add_argument('--type', dest='dataset_type', required=True, choices=['intra', 'inter'], help='dataset: "intra" or "inter"')
    parser.add_argument('--epochs', '-e', dest='max_epoch', type=int, default=20, help='max epoch')
    parser.add_argument('--emb_type', dest='emb_type', required=True, choices=['Word2Vec', 'Word2VecWiki','FastText', 'ELMo', 'Random', 'ELMoForManyLangs', 'None'], help='word embedding type')
    parser.add_argument('--emb_path', dest='emb_path', help='word embedding path')
    parser.add_argument('--emb_requires_grad_false', dest='emb_requires_grad', action='store_false', help='fixed word embedding or not')
    parser.add_argument('--emb_dim', type=int, default=200, help='word embedding dimension')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1, help='GPU ID for execution')
    parser.add_argument('--batch', '-b', dest='batch_size', type=int, default=16, help='mini batch size')
    parser.add_argument('--case', '-c', dest='case', type=str, required=True, choices=['ga', 'o', 'ni'], help='target "case" type')
    parser.add_argument('--media', '-m', dest='media', nargs='+', type=str, default=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], choices=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], help='training media type')
    parser.add_argument('--dump_dir', dest='dump_dir', type=str, required=True, help='model dump directory path')
    return parser

def initialize_model(gpu, vocab_size, v_vec, emb_requires_grad, args):
    emb_dim = args.emb_dim
    h_dim = None
    class_num = 2
    is_gpu = True
    if gpu == -1:
        is_gpu = False
    if args.emb_type == 'ELMo' or args.emb_type == 'ELMoForManyLangs':
        bilstm = BiLSTM(emb_dim, h_dim, class_num, vocab_size, is_gpu, v_vec, emb_type=args.emb_type, elmo_model_dir=args.emb_path)
    elif args.emb_type == 'None':
        bilstm = BiLSTM(emb_dim, h_dim, class_num, vocab_size, is_gpu, v_vec, emb_type=args.emb_type)
    else:
        bilstm = BiLSTM(emb_dim, h_dim, class_num, vocab_size, is_gpu, v_vec, emb_type=args.emb_type)
    if is_gpu:
        bilstm = bilstm.cuda()

    for m in bilstm.modules():
        print(m.__class__.__name__)
        weights_init(m)

    if args.emb_type != 'ELMo' and args.emb_type != 'ELMoForManyLangs' and args.emb_type != 'None':
        for param in bilstm.word_embed.parameters():
            param.requires_grad = emb_requires_grad

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

def translate_batch(batch, gpu, case, emb_type):
    x = batch[:, 0]
    y = batch[:, 1]
    files = batch[:, 2]
    batchsize = len(batch)

    max_length = x[0].shape[0]
    sentences = [i['単語'].values[4:] for i in batch[:, 0]]
    sentences = np.array(sentences)
    if emb_type == 'ELMo':
        x_wordID = elmo.batch_to_ids(sentences)
        if gpu >= 0:
            x_wordID = x_wordID.cuda()
    elif emb_type == 'ELMoForManyLangs':
        x_wordID = sentences
    else:
        x_wordID = translate_df_tensor(x, ['単語ID'], gpu)
        x_wordID = x_wordID.reshape(batchsize, -1)
    x_feature_emb_list = []
    for i in range(6):
        x_feature_emb = translate_df_tensor(x, [f'形態素{i}'], gpu)
        x_feature_emb = x_feature_emb.reshape(batchsize, -1)
        x_feature_emb_list.append(x_feature_emb)
    x_feature = translate_df_tensor(x, ['n単語目', 'n文節目','is主辞', 'is機能語','is_target_verb', '述語からの距離'], gpu)
    x = [x_wordID, x_feature_emb_list, x_feature]

    y = translate_df_y(y, case, -1)
    y = y.reshape(batchsize)
    y = torch.eye(max_length, dtype=torch.long)[y]
    if gpu >= 0:
        y = y.cuda()

    return x, y, files

def train(trains, vals, bilstm, args):
    print('--- start training ---')
    epochs = args.max_epoch+1
    lr = 0.001 #学習係数
    results = {}
    optimizer = optim.Adam(bilstm.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs):
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
            #0 paddingするために，長さで降順にソートする．
            argsort_index = np.array([i.shape[0] for i in batch[:, 0]]).argsort()[::-1]
            batch = batch[argsort_index]
            x, y, _ = translate_batch(batch, args.gpu, args.case, args.emb_type)
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

        print(f'[epoch: {epoch}]\tloss: {running_loss/(running_samples/args.batch_size)}\tacc: {running_correct/running_samples}')
        _results, _ = test(vals, bilstm, args)
        results[epoch] = _results
        save_model(epoch, bilstm, args.dump_dir, args.gpu)
    dump_dic(results, args.dump_dir, 'training_logs.json')
    best_epochs = defaultdict(lambda: defaultdict(float))
    for epoch in results:
        for domain in sorted(results[epoch].keys()):
            if results[epoch][domain]['F1']['F1-score']['total'] > best_epochs[domain]['F1-score(total)']:
                best_epochs[domain]['F1-score(total)'] = results[epoch][domain]['F1']['F1-score']['total']
                best_epochs[domain]['acc'] = results[epoch][domain]['acc']
                best_epochs[domain]['epoch'] = epoch
    dump_dic(best_epochs, args.dump_dir, 'training_result.json')
    print('--- finish training ---\n--- best F1-score epoch for each domain ---')
    for domain in sorted(best_epochs.keys()):
        print(f'{domain} [epoch: {best_epochs[domain]["epoch"]}]\tF1-score: {best_epochs[domain]["F1-score(total)"]}\tacc: {best_epochs[domain]["acc"]}')

def initialize_confusion_matrix():
    case_types = ['none', 'exo1', 'exo2', 'exoX', 'intra(dep)', 'intra(zero)', 'inter(zero)']
    index = pd.MultiIndex.from_arrays([['predicted']*7, case_types])
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
        if sum(confusion_matrix.loc['predicted', case_type]) != 0:
            df['precision'][case_type] = tp/sum(confusion_matrix.loc['predicted', case_type])
        #recall
        if sum(confusion_matrix['actual', case_type]) != 0:
            df['recall'][case_type] = tp/sum(confusion_matrix['actual', case_type])
        #F1-score
        if (df['precision'][case_type]+df['recall'][case_type]) != 0:
            df['F1-score'][case_type] = (2*df['precision'][case_type]*df['recall'][case_type])/(df['precision'][case_type]+df['recall'][case_type])
    all_tp = 0
    all_fp = 0
    all_fn = 0
    for case_type in case_types:
        all_tp += confusion_matrix['actual', case_type]['predicted', case_type]
        all_fp += sum(confusion_matrix.loc['predicted', case_type])
        all_fn += sum(confusion_matrix['actual', case_type])
    df.loc['total'] = 0
    df['precision']['total'] = all_tp/(all_tp+all_fp)
    df['recall']['total'] = all_tp/(all_tp+all_fn)
    df['F1-score']['total'] = (2*df['precision']['total']*df['recall']['total'])/(df['precision']['total']+df['recall']['total'])
    return df

def predicted_log(batch, pred, target_case, dump_dir, corrects):
    batchsize = len(batch)
    for i in range(batchsize):
        target_verb_index = batch[i][1].name
        predicted_argument_index = pred[i].item()
        actual_argument_index = int(batch[i][1][target_case].split(',')[0])
        target_verb = batch[i][0]['単語'][target_verb_index]
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


def test(tests, bilstm, args):
    results = defaultdict(lambda: defaultdict(float))
    logs = defaultdict(list)
    for domain in args.media:
        results[domain]['confusion_matrix'] = initialize_confusion_matrix()
    results['All']['confusion_matrix'] = initialize_confusion_matrix()

    bilstm.eval()
    criterion = nn.CrossEntropyLoss()
    N = len(tests)
    for i in tqdm(range(0, N, args.batch_size)):
        batch = tests[i:i+args.batch_size]
        batchsize = len(batch)

        #0 paddingするために，長さで降順にソートする．
        argsort_index = np.array([i.shape[0] for i in batch[:, 0]]).argsort()[::-1]
        batch = batch[argsort_index]
        x, y, files = translate_batch(batch, args.gpu, args.case, args.emb_type)

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
        # FOR debug
        case_types = ['none', 'exo1', 'exo2', 'exoX', 'intra(dep)', 'intra(zero)', 'inter(zero)']
        correct_of_confusion_matrix = 0
        for case_type in case_types:
            correct_of_confusion_matrix += results[domain]['confusion_matrix']['actual'][case_type]['predicted'][case_type]
        if correct_of_confusion_matrix != np.array(corrects).sum():
            import ipdb; ipdb.set_trace();
        for domain, log in predicted_log(batch, pred, args.case, args.dump_dir, corrects):
            logs[domain].append(log)

    for domain in args.media:
        results['All']['loss'] += results[domain]['loss']
        results['All']['samples'] += results[domain]['samples']
        results['All']['correct'] += results[domain]['correct']
        for i in range(results[domain]['confusion_matrix'].shape[0]):
            for j in range(results[domain]['confusion_matrix'].shape[1]):
                results['All']['confusion_matrix'].iat[i, j] += results[domain]['confusion_matrix'].iat[i, j]
        results[domain]['loss'] /= results[domain]['samples']
        results[domain]['acc'] = results[domain]['correct']/results[domain]['samples']
        results[domain]['F1'] = calculate_f1(results[domain]['confusion_matrix'])
    results['All']['loss'] /= results['All']['samples']
    results['All']['acc'] = results['All']['correct']/results['All']['samples']
    results['All']['F1'] = calculate_f1(results['All']['confusion_matrix'])
    for domain in sorted(results.keys()):
        print(f'[domain: {domain}]\ttest loss: {results[domain]["loss"]}\tF1-score: {results[domain]["F1"]["F1-score"]["total"]}\tacc: {results[domain]["acc"]}')
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

    wv = WordVector(args.emb_type, args.emb_path)
    is_intra = True
    if args.dataset_type == 'inter':
        is_intra = False
    datasets = load_datasets(wv, is_intra, args.media)
    trains, vals, tests = split(datasets)
    args.__dict__['trains_size'] = len(trains)
    args.__dict__['vals_size'] = len(vals)
    args.__dict__['tests_size'] = len(tests)

    bilstm = initialize_model(args.gpu, vocab_size=len(wv.index2word), v_vec= wv.vectors, emb_requires_grad=args.emb_requires_grad, args=args)
    dump_dic(args.__dict__, args.dump_dir, 'args.json')
    pprint(args.__dict__)
    train(trains, vals, bilstm, args)
    # train_loader = data.DataLoader(trains, batch_size=16, shuffle=True)
    # vals_loader = data.DataLoader(vals, batch_size=16, shuffle=True)

if __name__ == '__main__':
    main()
