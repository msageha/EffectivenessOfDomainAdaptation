import argparse
import json
from pprint import pprint
import torch

import sys
sys.path.append('../utils')

from loader import WordVector, load_datasets, split
from model import BiLSTM
from train import test, initialize_model, translate_df_tensor, translate_batch, dump_dic

def create_arg_parser():
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1, help='GPU ID for execution')
    parser.add_argument('--load_dir', dest='load_dir', type=str, required=True, help='model load directory path')
    return parser

def load_config(args):
    # "intra/Word2Vec_Fix/All_PN/ga" --emb_requires_grad_false
    with open(f'{args.load_dir}/args.json') as f:
        params = json.load(f)
    for key in params:
        if key=='gpu':
            continue
        args.__dict__[key] = params[key]

def load_model(epoch, bilstm, dump_dir, gpu):
    print('--- load model ---')
    bilstm.load_state_dict(torch.load(f'./{dump_dir}/model/{epoch}.pkl'))
    if gpu>=0:
        bilstm.cuda()

def max_f1_epochs_of_vals(train_result_path):
    with open(f'{train_result_path}/training_result.json') as f:
        val_results = json.load(f)
    return val_results

def dump_predict_logs(logs, dump_dir):
    os.makedirs(f'./{dump_dir}/predicts/', exist_ok=True)
    for domain in logs.keys():
        with open(f'{dump_dir}/predicts/{domain}.tsv', 'w') as f:
            f.write('\t'.join(log.keys()))
            f.write('\n')
            for log in logs[domain]:
                values = [str(value) for value in log.values()]
                f.write('\t'.join(values))
                f.write('\n')

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    load_config(args)

    wv = WordVector(args.emb_type, args.emb_path)
    is_intra = True
    if args.dataset_type == 'inter':
        is_intra = False
    datasets = load_datasets(wv, is_intra, args.media)
    _, _, tests = split(datasets)

    bilstm = initialize_model(args.gpu, vocab_size=len(wv.index2word), v_vec= wv.vectors, emb_requires_grad=args.emb_requires_grad, args=args)

    pprint(args.__dict__)
    val_results = max_f1_epochs_of_vals(args.load_dir)
    results = {}
    logs = {}
    domain = 'All'
    epoch = val_results[domain]['epoch']
    load_model(epoch, bilstm, args.load_dir, args.gpu)
    _results, _logs = test(tests, bilstm, args)
    results[domain] = _results[domain]
    results[domain]['epoch'] = epoch
    logs[domain] = _logs[domain]
    for domain in args.media:
        epoch = val_results[domain]['epoch']
        load_model(epoch, bilstm, args.load_dir, args.gpu)
        _results = test(tests, bilstm, args)
        results[domain] = _results[domain]
        results[domain]['epoch'] = epoch
        logs[domain] = _logs[domain]
    dump_dic(results, args.load_dir, 'test_logs.json')
    dump_predict_logs(logs, args.load_dir)

if __name__ == '__main__':
    main()
