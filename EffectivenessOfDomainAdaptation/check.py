import argparse

import sys
sys.path.append('../utils')
from loader import DatasetLoading


def create_arg_parser():
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--media', '-m', dest='media', nargs='+', type=str, default=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], choices=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], help='training media type')
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    emb_type = 'Word2VecWiki'
    emb_path = '../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt'
    dl = DatasetLoading(emb_type, emb_path, exo1_word='僕', exo2_word='おまえ', exoX_word='これ')
    dl.making_intra_df()

    trains_dict, vals_dict, tests_dict = dl.split_each_domain('intra')
    media = list(trains_dict.keys())
    for domain in media:
        if domain not in args.media:
            del trains_dict[domain]

    args.__dict__['trains_size'] = sum([len(trains_dict[domain]) for domain in trains_dict.keys()])
    args.__dict__['vals_size'] = sum([len(vals_dict[domain]) for domain in vals_dict.keys()])
    args.__dict__['tests_size'] = sum([len(tests_dict[domain]) for domain in tests_dict.keys()])

    print(f'training domain is {args.media}')
    print(f'trains_size {args.__dict__["trains_size"]}')
    print(f'vals_size {args.__dict__["vals_size"]}')
    print(f'tests_size {args.__dict__["tests_size"]}')


if __name__ == '__main__':
    main()