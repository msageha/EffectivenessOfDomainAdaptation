import pickle
import gensim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import re

domain_dict = {'PM':'雑誌','PN':'新聞', 'OW':'白書', 'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'PB':'書籍'}

class WordVector():
    def __init__(self, model):
        self.index2word = model.wv.index2word.copy()
        self.word2index = {key:model.wv.vocab[key].index for key in model.wv.vocab}
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

        self.vectors = np.vstack((self.vectors, unk_vector, none_vector, exo1_vector, exo2_vector, exoX_vector))

class FeatureToEmbedID:
    def __init__(self):
        feature_size_dict = {"feature:0":24, "feature:1":25, "feature:2":11, "feature:3":5, "feature:4":93,
          "feature:5":31, "feature:6":30119, "feature:7":35418, "feature:8":1,
          "feature:9":1, "feature:10":5545, "feature:11":1, "feature:12":7,
          "feature:13":1, "feature:14":5, "feature:15":1, "feature:16":1 }

        self.feature0 = {'': 0, "助詞":1, "未知語":2, "URL":3, "言いよどみ":4, "連体詞":5, "ローマ字文":6, "web誤脱":7,
          "英単語":8, "接頭辞":9, "助動詞":10, "接尾辞":11, "記号":12, "動詞":13, "漢文":14, "副詞":15, "形容詞":16,
          "接続詞":17, "補助記号":18, "代名詞":19, "名詞":20, "形状詞":21, "空白":22, "感動詞":23}

        self.feature1 = {"":0, "ＡＡ":1, "形状詞的":2, "一般":3, "括弧閉":4, "終助詞":5, "フィラー":6, "係助詞":7, "句点":8,
          "普通名詞":9, "数詞":10, "固有名詞":11, "準体助詞":12, "タリ":13, "括弧開":14, "読点":15, "形容詞的":16,
          "動詞的":17, "名詞的":18, "格助詞":19, "接続助詞":20, "助動詞語幹":21, "非自立可能":22, "文字":23, "副助詞":24}

        self.feature2 = {"":0, "助数詞可能":1, "一般":2, "副詞可能":3, "人名":4, "サ変形状詞可能":5, "顔文字":6,
          "助数詞":7, "地名":8, "サ変可能":9, "形状詞可能":10}

        self.feature3 = {"":0, "国":1, "名":2, "姓":3, "一般":4}

        self.feature4 = {"":0, "サ行変格":1, "文語助動詞-ヌ":2, "文語下二段-サ行":3, "文語下二段-ラ行":4, "下一段-バ行":5,
          "下一段-サ行":6, "文語四段-タ行":7, "助動詞-ヌ":8, "文語サ行変格":9, "下一段-ザ行":10, "文語助動詞-タリ-完了":11,
          "文語助動詞-ゴトシ":12, "下一段-カ行":13, "助動詞-レル":14, "文語助動詞-ナリ-断定":15, "文語ラ行変格":16,
          "文語四段-ハ行":17, "下一段-ガ行":18, "形容詞":19, "五段-バ行":20, "下一段-ナ行":21, "助動詞-ラシイ":22,
          "文語助動詞-ズ":23, "助動詞-ナイ":24, "五段-サ行":25, "五段-タ行":26, "文語助動詞-ケリ":27, "助動詞-ダ":28,
          "文語上一段-ナ行":29, "文語四段-マ行":30, "上一段-マ行":31, "文語下二段-ダ行":32, "文語助動詞-キ":33,
          "文語上一段-マ行":34, "文語助動詞-ベシ":35, "文語助動詞-ナリ-伝聞":36, "助動詞-ナンダ":37, "上一段-バ行":38,
          "助動詞-ジャ":39, "文語形容詞-ク":40, "文語上二段-ダ行":41, "文語下二段-タ行":42, "文語助動詞-タリ-断定":43,
          "文語下二段-ハ行":44, "文語四段-ガ行":45, "文語下二段-マ行":46, "文語助動詞-リ":47, "無変化型":48, "助動詞-ヘン":49,
          "文語下二段-ナ行":50, "上一段-ア行":51, "上一段-ガ行":52, "助動詞-デス":53, "五段-カ行":54, "助動詞-タ":55,
          "上一段-ザ行":56, "助動詞-タイ":57, "カ行変格":58, "五段-ガ行":59, "五段-ナ行":60, "文語上二段-バ行":61,
          "助動詞-ヤス":62, "五段-ワア行":63, "上一段-ラ行":64, "文語助動詞-ム":65, "上一段-ナ行":66, "五段-マ行":67,
          "文語形容詞-シク":68, "五段-ラ行":69, "文語四段-ラ行":70, "下一段-ラ行":71, "文語四段-サ行":72, "文語四段-カ行":73,
          "文語助動詞-ラシ":74, "助動詞-ヤ":75, "文語下一段-カ行":76, "助動詞-マイ":77, "文語下二段-ガ行":78, "助動詞-マス":79,
          "文語助動詞-マジ":80, "文語カ行変格":81, "下一段-タ行":82, "下一段-ダ行":83, "上一段-カ行":84, "文語上二段-ハ行":85,
          "下一段-ハ行":86, "文語助動詞-ジ":87, "上一段-タ行":88, "下一段-マ行":89, "文語下二段-カ行":90, "文語下二段-ア行":91,
          "下一段-ア行":92}

        self.feature5 = {"":0, "連用形-イ音便":1, "連体形-撥音便":2, "連用形-一般":3, "語幹-一般":4, "ク語法":5, "終止形-融合":6,
          "未然形-サ":7, "終止形-一般":8, "語幹-サ":9, "已然形-一般":10, "未然形-撥音便":11, "仮定形-一般":12, "連体形-一般":13,
          "連体形-省略":14, "未然形-補助":15, "連用形-ニ":16, "仮定形-融合":17, "終止形-促音便":18, "終止形-ウ音便":19,
          "未然形-一般":20, "連用形-促音便":21, "終止形-撥音便":22, "未然形-セ":23, "意志推量形":24, "命令形":25, "連用形-省略":26,
          "連用形-撥音便":27, "連用形-ウ音便":28, "連体形-補助":29, "連用形-融合":30}

class VirtualWordsDataFrame():
    def __init__(self):
        self.__exo1__()
        self.__exo2__()
        self.__exoX__()
        self.__none__()
        df = pd.DataFrame(columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'id', 'ga', 'ga_dep', 'o', 'o_dep', 'ni', 'ni_dep', 'type', 'n文節目', 'is主辞', 'n文目', 'is文末'])
        self.virtual_words = pd.concat([df, self.none, self.exoX, self.exo2, self.exo1], ignore_index=True, sort=False)

    def __exo1__(self):
        df = pd.DataFrame([[-1, '<exo1>', '代名詞', '', '', '', '', '', -1]], columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'n文節目'])
        self.exo1 = df

    def __exo2__(self):
        df = pd.DataFrame([[-2, '<exo2>', '代名詞', '', '', '', '', '', -1]], columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'n文節目'])
        self.exo2 = df

    def __exoX__(self):
        df = pd.DataFrame([[-3, '<exoX>', '代名詞', '', '', '', '', '', -1]], columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'n文節目'])
        self.exoX = df

    def __none__(self):
        df = pd.DataFrame([[-4, '<none>', '', '', '', '', '', '', -1]], columns=['n単語目', '単語', '形態素0', '形態素1', '形態素2', '形態素3', '形態素4', '形態素5', 'n文節目'])
        self.none = df

def is_num(text):
    m = re.match('\A[0-9]+\Z', text)
    if m: return True
    else: return False

def load_w2v(path):
    print(f'start loading word2vec from {path}')
    model = gensim.models.KeyedVectors.load(path)
    return model

def to_intra_sentential_df(df):
    last_sentence_indices = df[df['is文末']==True]
    start = 0
    for index in last_sentence_indices:
        end = index
        yield df.loc[start:end]
        start = index + 1

def case_tags(df, y, case):
    if is_num(y[case]):
        if (df['id'] == y[case]).sum():
            return (df['id'] == y[case]).idxmax()
        else:
            return 1 #文間ゼロ照応
    elif y[case] == 'exog':
        return 1
    elif y[case] == 'exo2':
        return 2
    elif y[case] == 'exo1':
        return 3
    else:
        return 0

def df_to_vector(df, wv):
    fe = FeatureToEmbedID()
    vwdf = VirtualWordsDataFrame()
    df = pd.concat([vwdf.virtual_words, df], ignore_index=True, sort=False)
    for index, row in df.iterrows():
        if row['単語'] in wv.word2index:
            row['単語'] = wv.word2index[row['単語']]
        else:
            row['単語'] = wv.word2index['<unk>']
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
            case = 'ga'
            y[case] = case_tags(df, y, case)
            case = 'o'
            y[case] = case_tags(df, y, case)
            case = 'ni'
            y[case] = case_tags(df, y, case)
            x = df.drop(labels=['id', 'ga', 'ga_dep', 'o', 'o_dep', 'ni', 'ni_dep', 'type', 'n文目', 'is文末'], axis=1)
            x = np.array(x)
            yield x, y

def load_datasets(path):
    model = load_w2v('../../data/embedding/Word2Vec/All.bin')
    wv = WordVector(model)
    with open(path, 'rb') as f:
        print(f'start loading datasets pickle from {path}')
        datasets = pickle.load(f)
    for domain in domain_dict:
        print(f'start making datasets in {domain}')
        for file in datasets:
            if domain in file:
                for x, y in df_to_vector(datasets[file], wv):
                    yield domain, x, y

def main():
    datasets = defaultdict(list)
    for domain, x, y in load_datasets('../datasets.pickle'):
        datasets[domain].append((x, y))
    return datasets