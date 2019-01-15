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

from model import BiLSTM, OneHot, FeatureAugmentation, ClassProbabilityShift
import train
import test

import sys
sys.path.append('../utils')
from loader import DatasetLoading, load_model
from store import dump_dict, save_model
from subfunc import return_file_domain, predicted_log
from calc_result import ConfusionMatrix


def create_arg_parser():
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--epochs', '-e', dest='max_epoch', type=int, default=15, help='max epoch')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1, help='GPU ID for execution')
    parser.add_argument('--load_dir', dest='load_dir', type=str, required=True, help='model load directory path')
    parser.add_argument('--model', dest='model', type=str, required=True, choices=['FT', 'MIX'])
    parser.add_argument('--dump_dir', dest='dump_dir')
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    load_config(args)

    emb_type = 'Word2VecWiki'

    dl = DatasetLoading(emb_type, args.emb_path, media=args.media)
    dl.making_intra_df()

    trains_dict, vals_dict, _ = dl.split_each_domain('intra')

    bilstm = train.initialize_model(args.gpu, vocab_size=len(dl.wv.index2word), v_vec=dl.wv.vectors, dropout_ratio=0.2, n_layers=3, model=args.model, statistics_of_each_case_type=None)

    pprint(args.__dict__)
    val_results = test.max_f1_epochs_of_vals(args.load_dir)

