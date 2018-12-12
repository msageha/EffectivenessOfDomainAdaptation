import numpy as np
import pandas as pd
import gensim
import re
import os
from collections import defaultdict
import pickle
import random

def get_tag_id(text):
    m = re.search(r'id="([0-9]+)"', text)
    if m: return m.group(1)
    return ''
def get_ga_tag(text):
    m = re.search(r'ga="(.+?)"', text)
    if m: return m.group(1)
    return ''
def get_o_tag(text):
    m = re.search(r'o="(.+?)"', text)
    if m: return m.group(1)
    return ''
def get_ni_tag(text):
    m = re.search(r' ni="(.+?)"', text)
    if m: return m.group(1)
    return ''
def get_ga_dep_tag(text):
    m = re.search(r'ga_dep="(.+?)"', text)
    if m: return m.group(1)
    return None
def get_o_dep_tag(text):
    m = re.search(r'o_dep="(.+?)"', text)
    if m: return m.group(1)
    return None
def get_ni_dep_tag(text):
    m = re.search(r' ni_dep="(.+?)"', text)
    if m: return m.group(1)
    return None
def is_num(text):
    m = re.match('\A[0-9]+\Z', text)
    if m: return True
    else: return False
def get_type(text):
    m = re.search(r'type="(.+?)"', text)
    if m: return m.group(1)
    return ''

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
    tag_id = get_tag_id(tags)
    ga_tag = get_ga_tag(tags)
    o_tag = get_o_tag(tags)
    ni_tag = get_ni_tag(tags)
    ga_dep_tag = get_ga_dep_tag(tags)
    o_dep_tag = get_o_dep_tag(tags)
    ni_dep_tag = get_ni_dep_tag(tags)
    verb_type = get_type(tags)
    df = pd.DataFrame([[word, pos_list[0], pos_list[1], pos_list[2], pos_list[3], pos_list[4], pos_list[5], tag_id, ga_tag, ga_dep_tag, o_tag, o_dep_tag, ni_tag, ni_dep_tag, verb_type]], columns=['単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'id', 'ga', 'ga_dep', 'o', 'o_dep', 'ni', 'ni_dep', 'type'])
    return df

def document_to_df(document):
    df = pd.DataFrame(columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'id', 'ga', 'ga_dep', 'o', 'o_dep', 'ni', 'ni_dep', 'type', 'n文節目', 'is主辞', 'n文目', 'is文末'])
    n_words = 0
    n_phrase = -1 #スタート時インクリメントされるため
    n_words_from_phrase = 0
    n_sentence = 0
    for i, line in enumerate(document.split('\n')):
        if line == '':
            continue
        elif line[0] == '*':
            head_word_number = int(line.split()[3].split('/')[0])
            n_phrase += 1
            n_words_from_phrase = 0
        elif line == 'EOS':
            n_sentence += 1
            n_words = 0
            n_phrase = -1
            i = df.columns.get_loc('is文末')
            df.iloc[-1, i] = True
        else:
            _df = line_to_df(line)
            is_head = False
            if n_words_from_phrase == head_word_number:
                is_head = True
            _df['n単語目'] = n_words
            _df['n文節目'] = n_phrase
            _df['is主辞'] = is_head
            _df['n文目'] = n_sentence
            _df['is文末'] = False
            n_words += 1 
            n_words_from_phrase += 1
            df = pd.concat([df, _df], ignore_index=True, sort=False)
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
    domain_dict = {'PM':'雑誌','PN':'新聞', 'OW':'白書', 'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'PB':'書籍'}
    domains = list(domain_dict.keys())
    main(dataset_path, domains)