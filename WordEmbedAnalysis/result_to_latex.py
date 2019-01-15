from glob import glob
import json
import pandas as pd
from collections import defaultdict

import sys
sys.path.append('../utils')
from calc_result import ConfusionMatrix
from store import dump_dict

# glob('./result/**/test_logs.json', recursive=True)


def load():
    for path in glob('./result/**/ga', recursive=True):
        path = '/'.join(path.split('/')[:-1])
        all_results = defaultdict(dict)
        for domain in ['All', 'OC', 'OY', 'OW', 'PB', 'PM', 'PN']:
            all_results[domain]['confusion_matrix'] = ConfusionMatrix()
        for case in ['ga', 'o', 'ni']:
            results = load_json(f'{path}/{case}/test_logs.json')
            for domain in results.keys():
                df = pd.DataFrame.from_dict(results[domain]['confusion_matrix'])
                for actual_key in ['exo1', 'exo2', 'exoX', 'inter(zero)', 'intra(dep)', 'intra(zero)', 'none']:
                    for predicted_key in  ['exo1', 'exo2', 'exoX', 'inter(zero)', 'inter(zero)_false', 'intra(dep)', 'intra(dep)_false', 'intra(zero)', 'intra(zero)_false', 'none']:
                        all_results[domain]['confusion_matrix'].df['actual'][actual_key]['predicted'][predicted_key] += df[f'actual_{actual_key}'][f'predicted_{predicted_key}']
        for domain in ['All', 'OC', 'OY', 'OW', 'PB', 'PM', 'PN']:
            all_results[domain]['F1'] = all_results[domain]['confusion_matrix'].calculate_f1().to_dict()
            all_results[domain]['confusion_matrix'] = all_results[domain]['confusion_matrix'].df.to_dict()

            tmp_dict1 = {}
            for key1 in all_results[domain]['confusion_matrix']:
                tmp_dict2 = {}
                for key2 in all_results[domain]['confusion_matrix'][key1]:
                    tmp_dict2['_'.join(key2)] = all_results[domain]['confusion_matrix'][key1][key2]
                tmp_dict1['_'.join(key1)] = tmp_dict2
            all_results[domain]['confusion_matrix'] = tmp_dict1
        dump_dict(all_results, f'{path}/all', 'test_logs')


def load_json(path):
    with open(path) as f:
        result = json.load(f)
    return result


def initialize_df_wiki():
    media = ['All', 'OC', 'OY', 'OW', 'PB', 'PM', 'PN']

    columns = pd.MultiIndex.from_arrays([['Word2Vec'] * 7, media])
    media = ['OC', 'OY', 'OW', 'PB', 'PM', 'PN', 'All']
    index = pd.MultiIndex.from_arrays([['target'] * 7, media])
    df = pd.DataFrame(data='-', index=index, columns=columns)


def wiki():
    # media = ['OC', 'OY', 'OW', 'PB', 'PM', 'PN', 'All']
    media = ['OC', 'OY', 'OW', 'PB', 'PM', 'PN', 'All', 'All_OC', 'All_OY', 'All_OW', 'All_PB', 'All_PM', 'All_PN', 'entity_vector']
    for domain in media:
        path= f'result/Word2Vec/{domain}/all/test_logs.json'
        result= load_json(path)
        add_result_to_pandas(result, domain, df)

def initialize_df():
    index_list = ['OC', 'OY', 'OW', 'PB', 'PM', 'PN', 'All']
    index = pd.MultiIndex.from_arrays([['target'] * 7, index_list])
    columns = ['None'] + ['Random'] + ['FastText'] + ['Word2Vec'] + ['Elmo(200)', 'Elmo(1024)']
    # columns = ['OC', 'OY', 'OW', 'PB', 'PM', 'PN', 'All', 'All_OC', 'All_OY', 'All_OW', 'All_PB', 'All_PM', 'All_PN', 'entity_vector']
    df = pd.DataFrame(data='-', index=index, columns=columns)

def add_result_to_pandas(result, column, df, case_type='total(intra)'):
    for domain in result.keys():
        f1= result[domain]['F1']['F1-score'][case_type]
        f1= '{:.2f}'.format(f1*100)
        df[column]['target', domain]= f1


path = 'ELMo/200/all/test_logs.json'
result= load_json(path)
add_result_to_pandas(result, 'Elmo(200)', df)
path = 'ELMo/1024/all/test_logs.json'
result= load_json(path)
add_result_to_pandas(result, 'Elmo(1024)', df)
path = 'None/all/test_logs.json'
result= load_json(path)
add_result_to_pandas(result, 'None', df)
path = 'Random/bccwj_intra_training/all/test_logs.json'
result= load_json(path)
add_result_to_pandas(result, 'Random', df)
path = 'FastText/All/all/test_logs.json'
result= load_json(path)
add_result_to_pandas(result, 'FastText', df)
path = 'Word2Vec/All/all/test_logs.json'
result= load_json(path)
add_result_to_pandas(result, 'Word2Vec', df)