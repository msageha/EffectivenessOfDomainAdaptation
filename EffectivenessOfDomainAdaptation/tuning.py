import argparse
from functools import partial
import json
import os
import optuna
from pprint import pprint
import torch

import sys
sys.path.append('../utils')
from loader import WordVector, load_datasets, split
from model import BiLSTM
from train import train, create_arg_parser, dump_dic, initialize_model

def tuning(trains, vals, wv, args, trial):
    # num of lstm layer
    n_layers = trial.suggest_int('n_layers', 1, 3)
    # dropout_rate
    dropout_ratio = trial.suggest_categorical('dropout_rate', [0, 0.1, 0.2, 0.3])

    bilstm = initialize_model(args.gpu, vocab_size=len(wv.index2word), v_vec= wv.vectors, dropout_ratio=dropout_ratio, n_layers=n_layers)

    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size=trial.suggest_categorical('batch_size', [16, 32, 64])
    F1 = train(trains, vals, bilstm, args, lr=lr, batch_size=batch_size)
    return 1 - F1

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    emb_type = 'Word2VecWiki'
    wv = WordVector(emb_type, args.emb_path)
    is_intra = True
    datasets = load_datasets(wv, is_intra, args.media)
    trains, vals, tests = split(datasets)
    args.__dict__['trains_size'] = len(trains)
    args.__dict__['vals_size'] = len(vals)
    args.__dict__['tests_size'] = len(tests)

    dump_dic(args.__dict__, args.dump_dir, 'args.json')
    pprint(args.__dict__)
    
    # 目的関数にデータを適用する
    f = partial(tuning, trains, vals, wv, args)
    study = optuna.create_study()
    study.optimize(f, n_trials=2)

    print("best params: ", study.best_params)
    print("best test accuracy: ", 1 - study.best_value)

if __name__ == '__main__':
    main()