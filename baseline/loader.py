from collections import defaultdict
import gensim
import numpy as np
import pandas as pd
import pickle
import re

class WordVector():
    def __init__(self, emb_type, path=None):
        if emb_type == 'Word2Vec' or emb_type == 'FastText':
            model = load_word_vector(path)
            self.index2word = ['padding'] + model.wv.index2word.copy()
            self.word2index = {word:i for i, word in enumerate(self.index2word)}
            padding_vector = np.zeros((1, 200))
            self.vectors = model.wv.vectors.copy()

            #UNK
            self.index2word.append('<unk>')
            self.word2index['<unk>'] = len(self.word2index)
            unk_vector = 2*np.random.rand(1, 200) - 1
            #none
            self.index2word.append('<none>')
            self.word2index['<none>'] = len(self.word2index)
            none_vector = np.zeros((1, 200))
            #exo1
            self.index2word.append('<exo1>')
            self.word2index['<exo1>'] = len(self.word2index)
            exo1_vector = model.wv.get_vector('私').reshape(1, 200)
            #exo2
            self.index2word.append('<exo2>')
            self.word2index['<exo2>'] = len(self.word2index)
            exo2_vector = model.wv.get_vector('あなた').reshape(1, 200)
            #exoX
            self.index2word.append('<exoX>')
            self.word2index['<exoX>'] = len(self.word2index)
            exoX_vector = model.wv.get_vector('これ').reshape(1, 200)
            self.vectors = np.vstack((padding_vector, self.vectors, unk_vector, none_vector, exo1_vector, exo2_vector, exoX_vector))
        elif emb_type == 'Random':
            self.index2word = ['padding']
            with open(path, 'r') as f:
                for line in f:
                    word = line.strip()
                    self.index2word.append(word)
            self.word2index = {word:i for i, word in enumerate(self.index2word)}
            self.vectors = None
        elif emb_type == 'ELMo':
            self.index2word = ['<unk>']
            self.word2index = {'<unk>':0}
            self.vectors = None
        else:
            print(f'unexpected emb_type: {emb_type}. Please check it.')
            return 0

class FeatureToEmbedID:
    def __init__(self):
        feature_size_dict = {"feature:0":24, "feature:1":25, "feature:2":11, "feature:3":5, "feature:4":93,
          "feature:5":31, "feature:6":30119, "feature:7":35418, "feature:8":1,
          "feature:9":1, "feature:10":5545, "feature:11":1, "feature:12":7,
          "feature:13":1, "feature:14":5, "feature:15":1, "feature:16":1 }

        self.feature0 = {'': 1, "助詞":2, "未知語":3, "URL":4, "言いよどみ":5, "連体詞":6, "ローマ字文":7, "web誤脱":8, "英単語":9, "接頭辞":10, "助動詞":11, "接尾辞":12, "記号":13, "動詞":14, "漢文":15, "副詞":16, "形容詞":17,
          "接続詞":18, "補助記号":19, "代名詞":20, "名詞":21, "形状詞":22, "空白":23, "感動詞":24}

        self.feature1 = {'': 1, 'ＡＡ': 2, '形状詞的': 3, '一般': 4, '括弧閉': 5, '終助詞': 6, 'フィラー': 7, '係助詞': 8, '句点': 9, '普通名詞': 10, '数詞': 11, '固有名詞': 12, '準体助詞': 13, 'タリ': 14, '括弧開': 15, '読点': 16, '形容詞的': 17, '動詞的': 18, '名詞的': 19, '格助詞': 20, '接続助詞': 21, '助動詞語幹': 22, '非自立可能': 23, '文字': 24, '副助詞': 25}

        self.feature2 = {'': 1, '助数詞可能': 2, '一般': 3, '副詞可能': 4, '人名': 5, 'サ変形状詞可能': 6, '顔文字': 7, '助数詞': 8, '地名': 9, 'サ変可能': 10, '形状詞可能': 11}

        self.feature3 = {"":1, "国":2, "名":3, "姓":4, "一般":5}

        self.feature4 = {'': 1, 'サ行変格': 2, '文語助動詞-ヌ': 3, '文語下二段-サ行': 4, '文語下二段-ラ行': 5, '下一段-バ行': 6, '下一段-サ行': 7, '文語四段-タ行': 8, '助動詞-ヌ': 9, '文語サ行変格': 10, '下一段-ザ行': 11, '文語助動詞-タリ-完了': 12, '文語助動詞-ゴトシ': 13, '下一段-カ行': 14, '助動詞-レル': 15, '文語助動詞-ナリ-断定': 16, '文語ラ行変格': 17, '文語四段-ハ行': 18, '下一段-ガ行': 19, '形容詞': 20, '五段-バ行': 21, '下一段-ナ行': 22, '助動詞-ラシイ': 23, '文語助動詞-ズ': 24, '助動詞-ナイ': 25, '五段-サ行': 26, '五段-タ行': 27, '文語助動詞-ケリ': 28, '助動詞-ダ': 29, '文語上一段-ナ行': 30, '文語四段-マ行': 31, '上一段-マ行': 32, '文語下二段-ダ行': 33, '文語助動詞-キ': 34, '文語上一段-マ行': 35, '文語助動詞-ベシ': 36, '文語助動詞-ナリ-伝聞': 37, '助動詞-ナンダ': 38, '上一段-バ行': 39, '助動詞-ジャ': 40, '文語形容詞-ク': 41, '文語上二段-ダ行': 42, '文語下二段-タ行': 43, '文語助動詞-タリ-断定': 44, '文語下二段-ハ行': 45, '文語四段-ガ行': 46, '文語下二段-マ行': 47, '文語助動詞-リ': 48, '無変化型': 49, '助動詞-ヘン': 50, '文語下二段-ナ行': 51, '上一段-ア行': 52, '上一段-ガ行': 53, '助動詞-デス': 54, '五段-カ行': 55, '助動詞-タ': 56, '上一段-ザ行': 57, '助動詞-タイ': 58, 'カ行変格': 59, '五段-ガ行': 60, '五段-ナ行': 61, '文語上二段-バ行': 62, '助動詞-ヤス': 63, '五段-ワア行': 64, '上一段-ラ行': 65, '文語助動詞-ム': 66, '上一段-ナ行': 67, '五段-マ行': 68, '文語形容詞-シク': 69, '五段-ラ行': 70, '文語四段-ラ行': 71, '下一段-ラ行': 72, '文語四段-サ行': 73, '文語四段-カ行': 74, '文語助動詞-ラシ': 75, '助動詞-ヤ': 76, '文語下一段-カ行': 77, '助動詞-マイ': 78, '文語下二段-ガ行': 79, '助動詞-マス': 80, '文語助動詞-マジ': 81, '文語カ行変格': 82, '下一段-タ行': 83, '下一段-ダ行': 84, '上一段-カ行': 85, '文語上二段-ハ行': 86, '下一段-ハ行': 87, '文語助動詞-ジ': 88, '上一段-タ行': 89, '下一段-マ行': 90, '文語下二段-カ行': 91, '文語下二段-ア行': 92, '下一段-ア行': 93}

        self.feature5 = {'': 1, '連用形-イ音便': 2, '連体形-撥音便': 3, '連用形-一般': 4, '語幹-一般': 5, 'ク語法': 6, '終止形-融合': 7, '未然形-サ': 8, '終止形-一般': 9, '語幹-サ': 10, '已然形-一般': 11, '未然形-撥音便': 12, '仮定形-一般': 13, '連体形-一般': 14, '連体形-省略': 15, '未然形-補助': 16, '連用形-ニ': 17, '仮定形-融合': 18, '終止形-促音便': 19, '終止形-ウ音便': 20, '未然形-一般': 21, '連用形-促音便': 22, '終止形-撥音便': 23, '未然形-セ': 24, '意志推量形': 25, '命令形': 26, '連用形-省略': 27, '連用形-撥音便': 28, '連用形-ウ音便': 29, '連体形-補助': 30, '連用形-融合': 31}

class VirtualWordsDataFrame():
    def __init__(self):
        self.__exo1__()
        self.__exo2__()
        self.__exoX__()
        self.__none__()
        df = pd.DataFrame(columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'id', 'ga', 'ga_dep', 'o', 'o_dep', 'ni', 'ni_dep', 'type', 'n文節目', 'is主辞', 'n文目', 'is文末'])
        self.virtual_words = pd.concat([df, self.none, self.exoX, self.exo2, self.exo1], ignore_index=True, sort=False)

    def __exo1__(self):
        df = pd.DataFrame([[-1, '<exo1>', '代名詞', '', '', '', '', '', -1, 1, -1]], columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'n文節目', 'is主辞', 'n文目'])
        self.exo1 = df

    def __exo2__(self):
        df = pd.DataFrame([[-2, '<exo2>', '代名詞', '', '', '', '', '', -1, 1, -1]], columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'n文節目', 'is主辞', 'n文目'])
        self.exo2 = df

    def __exoX__(self):
        df = pd.DataFrame([[-3, '<exoX>', '代名詞', '', '', '', '', '', -1, 1, -1]], columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'n文節目', 'is主辞', 'n文目'])
        self.exoX = df

    def __none__(self):
        df = pd.DataFrame([[-4, '<none>', '', '', '', '', '', '', -1, 1, -1]], columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'n文節目', 'is主辞', 'n文目'])
        self.none = df

def is_num(text):
    m = re.match('\A[0-9]+\Z', text)
    if m: return True
    else: return False

def load_word_vector(path):
    print(f'--- start loading Word Vector from {path} ---')
    model = gensim.models.KeyedVectors.load(path)
    return model

def to_intra_sentential_df(df):
    last_sentence_indices = df['is文末'][df['is文末']==True].index
    start = 0
    for index in last_sentence_indices:
        end = index
        yield df.loc[start:end]
        start = index + 1

def case_tags_to_id(df, y, case):
    sentence_start_id = 4
    sentence_end_id = 4
    for index in df[df['is文末']==True].index:
        if index < y.name:
            sentence_start_id = index+1
        if index >= y.name:
            sentence_end_id = index
    if is_num(y[case]):
        if (df['id'] == y[case]).sum():
            if sentence_start_id <= (df['id'] == y[case]).idxmax() and (df['id'] == y[case]).idxmax() <= sentence_end_id:
                return (df['id'] == y[case]).idxmax(), 'intra_'
            else:
                return (df['id'] == y[case]).idxmax(), 'inter_'
        else:
            return 1, 'inter_' #文内ゼロの場合，文間ゼロ照応はexogと同じタグIDに．
    elif y[case] == 'exog':
        return 1, 'exog'
    elif y[case] == 'exo2':
        return 2, 'exo2'
    elif y[case] == 'exo1':
        return 3, 'exo1'
    else:
        return 0, 'none'

def df_to_intra_vector(df, wv):
    fe = FeatureToEmbedID()
    vwdf = VirtualWordsDataFrame()
    df = pd.concat([vwdf.virtual_words, df], ignore_index=True, sort=False)
    df['単語ID'] = df['単語']
    for index, row in df.iterrows():
        if row['単語'] in wv.word2index:
            row['単語ID'] = wv.word2index[row['単語']]
        else:
            row['単語ID'] = wv.word2index['<unk>']
        row['形態素0'] = fe.feature0[row['形態素0']]
        row['形態素1'] = fe.feature1[row['形態素1']]
        row['形態素2'] = fe.feature2[row['形態素2']]
        row['形態素3'] = fe.feature3[row['形態素3']]
        row['形態素4'] = fe.feature4[row['形態素4']]
        row['形態素5'] = fe.feature5[row['形態素5']]
        if row['is主辞']:
            row['is主辞'] = 1
        else:
            row['is主辞'] = 0
    for index, row in df.iterrows():
        if row['type'] == 'noun' or row['type'] == 'pred':
            y = row.loc[['ga', 'ga_dep', 'o', 'o_dep', 'ni', 'ni_dep', 'type']].copy()
            cases = ['ga', 'o', 'ni']
            for case in cases:
                y[case], tag = case_tags_to_id(df, y, case)
                if y[f'{case}_dep'] == None:
                    y[f'{case}_dep'] = tag
                else:
                    y[f'{case}_dep'] = tag + y[f'{case}_dep']
            x = df.drop(labels=['id', 'ga', 'ga_dep', 'o', 'o_dep', 'ni', 'ni_dep', 'type', 'n文目', 'is文末'], axis=1).copy()
            x['is_target_verb'] = 0
            i = x.columns.get_loc('is_target_verb')
            x.iloc[index, i] = 1
            x['述語からの距離'] = x.index - index
            yield x, y

def df_to_inter_vector(df, wv):
    fe = FeatureToEmbedID()
    vwdf = VirtualWordsDataFrame()
    df = pd.concat([vwdf.virtual_words, df], ignore_index=True, sort=False)
    df['単語ID'] = df['単語']
    for index, row in df.iterrows():
        if row['単語'] in wv.word2index:
            row['単語ID'] = wv.word2index[row['単語']]
        else:
            row['単語ID'] = wv.word2index['<unk>']
        row['形態素0'] = fe.feature0[row['形態素0']]
        row['形態素1'] = fe.feature1[row['形態素1']]
        row['形態素2'] = fe.feature2[row['形態素2']]
        row['形態素3'] = fe.feature3[row['形態素3']]
        row['形態素4'] = fe.feature4[row['形態素4']]
        row['形態素5'] = fe.feature5[row['形態素5']]
        if row['is主辞']:
            row['is主辞'] = 1
        else:
            row['is主辞'] = 0
    for index, row in df.iterrows():
        if row['type'] == 'noun' or row['type'] == 'pred':
            y = row.loc[['ga', 'ga_dep', 'o', 'o_dep', 'ni', 'ni_dep', 'type']].copy()
            cases = ['ga', 'o', 'ni']
            for case in cases:
                y[case], tag = case_tags_to_id(df, y, case)
                if y[f'{case}_dep'] == None:
                    y[f'{case}_dep'] = tag
                else:
                    y[f'{case}_dep'] = tag + y[f'{case}_dep']
            x = df.drop(labels=['id', 'ga', 'ga_dep', 'o', 'o_dep', 'ni', 'ni_dep', 'type'], axis=1).copy()
            x['is_target_verb'] = 0
            i = x.columns.get_loc('is_target_verb')
            x.iloc[index, i] = 1
            x['述語からの距離'] = x.index - index
            yield x, y

def creating_datasets_for_each_domain(path, wv, is_intra, media):
    with open(path, 'rb') as f:
        print(f'--- start loading datasets pickle from {path} ---')
        datasets = pickle.load(f)
    for domain in media: #メディアごとに処理を行う
        print(f'--- start making datasets in {domain} ---')
        for file in datasets:
            if domain in file:
                if is_intra:
                    for intra_df in to_intra_sentential_df(datasets[file]):
                        for x, y in df_to_intra_vector(intra_df, wv):
                            yield domain, x, y, file
                else:
                    for x, y in df_to_inter_vector(datasets[file], wv):
                        yield domain, x, y, file

def load_datasets(wv, is_intra, media=['OC', 'OY', 'OW', 'PB', 'PM', 'PN'], pickle_path='../datasets.pickle'):
    datasets = defaultdict(list)
    for domain, x, y, file in creating_datasets_for_each_domain(pickle_path, wv, is_intra, media):
        datasets[domain].append((x, y, file))
    return datasets

def split(dataset, sizes=[0.7, 0.1, 0.2]):
    trains = []
    vals = []
    tests = []
    for domain in dataset.keys():
        dataset_size = len(dataset[domain])
        end = int(sizes[0]/sum(sizes)*dataset_size)
        train = dataset[domain][:end]
        start = end
        end += int(sizes[1]/sum(sizes)*dataset_size)
        val = dataset[domain][start:end]
        start = end
        test = dataset[domain][start:]
        trains += train
        vals += val
        tests += test
    trains = np.array(trains)
    vals = np.array(vals)
    tests = np.array(tests)
    return trains, vals, tests