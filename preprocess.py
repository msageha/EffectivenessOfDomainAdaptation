import numpy as np
import pandas as pd
import gensim
import re
import os
from collections import defaultdict
import pickle
import random

from utils.regular_expression import is_num, extraction_num


def get_id_tag(text):
    m = re.search(r'id="([0-9]+)"', text)
    if m:
        return m.group(1)
    return ''


def get_eq_tag(text):
    m = re.search(r'eq="([0-9]+)"', text)
    if m:
        return m.group(1)
    return ''


def get_type_tag(text):
    m = re.search(r'type="(.+?)"', text)
    if m:
        return m.group(1)
    return ''


def get_ga_tag(text):
    ids = []
    for m in re.finditer(r'ga="(.+?)"', text):
        if m.group(1) != 'ana_cla':
            ids.append(m.group(1))
    return ','.join(ids)


def get_o_tag(text):
    ids = []
    for m in re.finditer(r'o="(.+?)"', text):
        if m.group(1) != 'ana_cla':
            ids.append(m.group(1))
    return ','.join(ids)


def get_ni_tag(text):
    ids = []
    for m in re.finditer(r' ni="(.+?)"', text):
        if m.group(1) != 'ana_cla':
            ids.append(m.group(1))
    return ','.join(ids)


# def get_ga_dep_tag(text):
#     m = re.search(r'ga_dep="(.+?)"', text)
#     if m: return m.group(1)
#     return None
# def get_o_dep_tag(text):
#     m = re.search(r'o_dep="(.+?)"', text)
#     if m: return m.group(1)
#     return None
# def get_ni_dep_tag(text):
#     m = re.search(r' ni_dep="(.+?)"', text)
#     if m: return m.group(1)
#     return None


def load_document(path):
    document = ''
    with open(path) as f:
        for line in f:
            if line[0] == '#':
                continue
            document += line
    return document


def line_to_df(line):
    word, pos, tags = line.split('\t')
    pos_list = pos.split(',')
    for i in range(len(pos_list)):
        if pos_list[i] == '*':
            pos_list[i] = ''
    id_tag = get_id_tag(tags)
    ga_tag = get_ga_tag(tags)
    o_tag = get_o_tag(tags)
    ni_tag = get_ni_tag(tags)
    eq_tag = get_eq_tag(tags)
    verb_type = get_type_tag(tags)
    df = pd.DataFrame(
        [
            [word, pos_list[0], pos_list[1], pos_list[2], pos_list[3], pos_list[4], pos_list[5],
             id_tag, eq_tag, ga_tag, o_tag, ni_tag, verb_type]
        ],
        columns=['単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5',
            'id', 'eq','ga', 'o', 'ni', 'verb_type'
        ]
    )
    return df


def decision_case_type(df, case_id, verb_index):
    if is_num(case_id):
        sentence_start_id = 0
        sentence_end_id = 0
        for index in df[df['is文末']==True].index:
            if index < verb_index:
                sentence_start_id = index+1
            if index >= verb_index:
                sentence_end_id = index
                break
        case_index = (df['id'] == case_id).idxmax()
        #文内照応
        if sentence_start_id <= case_index and case_index <= sentence_end_id:
            verb_phrase_num = df['n文節目'][verb_index]
            dependency_relation_phrase = df['係り先文節'][case_index]
            dependency_relation_phrase = extraction_num(dependency_relation_phrase)
            if verb_phrase_num == dependency_relation_phrase:
                return 'intra(dep)'
            else:
                return 'intra(zero)'
        #文間照応
        else:
            return 'inter(zero)'
    #外界照応or照応なし
    elif case_id == 'exo1':
        return 'exo1'
    elif case_id == 'exo2':
        return 'exo2'
    elif case_id == 'exog':
        return 'exoX'
    elif case_id == '':
        return 'none'
    else:
        print('Error!!!')


def search_verbs(df):
    for index, row in df.iterrows():
        if row['verb_type'] == 'pred' or row['verb_type'] == 'noun':
            for case in ['ga', 'o', 'ni']:
                case_types = []
                case_ids = row[case].split(',')
                for case_id in case_ids:
                    case_type = decision_case_type(df, case_id, index)
                    case_types.append(case_type)
                order = [
                    'intra(dep)',
                    'intra(zero)',
                    'inter(zero)',
                    'exo1',
                    'exo2',
                    'exoX',
                    'none',
                ]
                order_arg_index = [order.index(case_type) for case_type in case_types]
                order_arg_index = np.array(order_arg_index).argsort()
                if len(order_arg_index) != len(case_types):
                    print('Error!!!')
                case_ids = [case_ids[i] for i in order_arg_index]
                case_types = [case_types[i] for i in order_arg_index]
                df[f'{case}'][index] = ','.join(case_ids)
                df[f'{case}_type'][index] = ','.join(case_types)


def document_to_df(document):
    df = pd.DataFrame(
        columns=['n単語目', '単語',
            '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5',
            'id', 'eq', 'ga', 'ga_type', 'o', 'o_type', 'ni', 'ni_type', 'verb_type',
            'n文節目', '係り先文節', 'is主辞', 'is機能語','n文目', 'is文末',
        ]
    )
    n_words = 0
    n_words_from_phrase = 0
    n_sentence = 0
    for i, line in enumerate(document.split('\n')):
        if line == '':
            continue
        elif line[0] == '*':
            n_phrase = int(line.split()[1])
            dependency_relation_phrase = line.split()[2]
            head_word_number = int(line.split()[3].split('/')[0])
            function_word_number = int(line.split()[3].split('/')[1])
            n_words_from_phrase = 0
        elif line == 'EOS':
            n_sentence += 1
            n_words = 0
            i = df.columns.get_loc('is文末')
            df.iloc[-1, i] = True
        else:
            _df = line_to_df(line)
            is_head = False
            if n_words_from_phrase == head_word_number:
                is_head = True
            is_function = False
            if n_words_from_phrase == function_word_number:
                is_function = True
            _df['n単語目'] = n_words
            _df['n文節目'] = n_phrase
            _df['係り先文節'] = dependency_relation_phrase
            _df['is主辞'] = is_head
            _df['is機能語'] = is_function
            _df['n文目'] = n_sentence
            _df['is文末'] = False
            n_words += 1 
            n_words_from_phrase += 1
            df = pd.concat(
                [df, _df],
                ignore_index=True, sort=False
            )
    search_verbs(df)
    return df


def main(path, domains):
    datasets = {}
    for domain in domains:
        print(domain)
        for file in os.listdir(f'{path}/{domain}'):
            document = load_document(f'{path}/{domain}/{file}')
            df = document_to_df(document)
            datasets[file] = df
    with open(f'./datasets.pickle', mode='wb') as f:
        pickle.dump(datasets, f)


if __name__ == '__main__':
    dataset_path = '../data/pas'
    domain_dict = {
        'OC' : 'Yahoo!知恵袋',
        'OW' : '白書',
        'OY' : 'Yahoo!ブログ',
        'PB' : '書籍',
        'PM' : '雑誌',
        'PN' : '新聞',
    }
    domains = list(domain_dict.keys())
    main(dataset_path, domains)