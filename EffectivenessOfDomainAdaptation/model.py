import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from allennlp.modules.elmo import Elmo, batch_to_ids

class BiLSTM(nn.Module):
    def __init__(self, v_size, v_vec, dropout_ratio, n_layers, emb_dim=200, n_labels=2, gpu=True, batch_first=True):
        super(BiLSTM, self).__init__()
        self.gpu = gpu
        self.h_dim = (emb_dim+34)//2
        self.word_embed = nn.Embedding(v_size, emb_dim, padding_idx=0)
        v_vec = torch.tensor(v_vec)
        self.word_embed.weight.data.copy_(v_vec)
        self.dropout_ratio = dropout_ratio
        self.n_layers = n_layers

        feature_embed_layers = []
        feature_embed_size = {
            "feature:0" : 25,
            "feature:1" : 26,
            "feature:2" : 12,
            "feature:3" : 6,
            "feature:4" : 94,
            "feature:5" : 32
        }
        for key in feature_embed_size:
            size = feature_embed_size[key]
            feature_embed = nn.Embedding(size, 5, padding_idx=0)
            feature_embed.weight.data[0] = torch.zeros(5)
            feature_embed_layers.append(feature_embed)
        self.feature_embed_layers = nn.ModuleList(feature_embed_layers)
        self.drop_target = nn.Dropout(p=dropout_ratio)

        lstm_layers = []
        for i in range(self.n_layers):
            lstm = nn.LSTM(input_size=emb_dim+34, hidden_size=self.h_dim, batch_first=batch_first, bidirectional=True)
            lstm_layers.append(lstm)
        self.lstm_layers = nn.ModuleList(lstm_layers)
        self.l1 = nn.Linear(self.h_dim*2, n_labels)

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, x):
        self.hidden = self.init_hidden(x[2].size(0))
        word_emb = self.word_embed(x[0])
        feature_emb_list = []
        for i, _x in enumerate(x[1]):
            feature_emb = self.feature_embed_layers[i](_x)
            feature_emb_list.append(feature_emb)
        x_feature = torch.tensor(x[2], dtype=torch.float, device=x[2].device)

        x = torch.cat(
            (word_emb, feature_emb_list[0], feature_emb_list[1], feature_emb_list[2], feature_emb_list[3], feature_emb_list[4], feature_emb_list[5], x_feature),
            dim=2
        )

        x = self.drop_target(x)
        for i in range(self.n_layers):
            x, hidden = self.lstm_layers[i](x, self.hidden)
        # out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]

        out = self.l1(x)
        return out

class OneHot(nn.Module):
    def __init__(self, v_size, v_vec, dropout_ratio, n_layers, emb_dim=200, n_labels=2, gpu=True, batch_first=True):
        super(OneHot, self).__init__()
        self.gpu = gpu
        self.h_dim = (emb_dim+40)//2
        self.word_embed = nn.Embedding(v_size, emb_dim, padding_idx=0)
        v_vec = torch.tensor(v_vec)
        self.word_embed.weight.data.copy_(v_vec)
        self.dropout_ratio = dropout_ratio
        self.n_layers = n_layers
        self.domain_label_encoder = {'OC':0, 'OY':1, 'OW':2, 'PB':3, 'PM':4, 'PN':5}

        feature_embed_layers = []
        feature_embed_size = {
            "feature:0" : 25,
            "feature:1" : 26,
            "feature:2" : 12,
            "feature:3" : 6,
            "feature:4" : 94,
            "feature:5" : 32
        }
        for key in feature_embed_size:
            size = feature_embed_size[key]
            feature_embed = nn.Embedding(size, 5, padding_idx=0)
            feature_embed.weight.data[0] = torch.zeros(5)
            feature_embed_layers.append(feature_embed)
        self.feature_embed_layers = nn.ModuleList(feature_embed_layers)
        self.drop_target = nn.Dropout(p=dropout_ratio)

        lstm_layers = []
        for i in range(self.n_layers):
            lstm = nn.LSTM(input_size=emb_dim+40, hidden_size=self.h_dim, batch_first=batch_first, bidirectional=True)
            lstm_layers.append(lstm)
        self.lstm_layers = nn.ModuleList(lstm_layers)
        self.l1 = nn.Linear(self.h_dim*2, n_labels)

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, x, domains):
        self.hidden = self.init_hidden(x[2].size(0))
        word_emb = self.word_embed(x[0])
        feature_emb_list = []
        for i, _x in enumerate(x[1]):
            feature_emb = self.feature_embed_layers[i](_x)
            feature_emb_list.append(feature_emb)
        x_feature = torch.tensor(x[2], dtype=torch.float, device=x[2].device)

        x_onehot = torch.zeros(x[0].size(0), x[0].size(1), 6)
        for i, domain in enumerate(domains):
            domain_index = self.domain_label_encoder[domain]
            x_onehot[i, :, domain_index] = 1
        if self.gpu:
            x_onehot = x_onehot.cuda()

        x = torch.cat(
            (word_emb, feature_emb_list[0], feature_emb_list[1], feature_emb_list[2], feature_emb_list[3], feature_emb_list[4], feature_emb_list[5], x_feature, x_onehot),
            dim=2
        )

        x = self.drop_target(x)
        for i in range(self.n_layers):
            x, hidden = self.lstm_layers[i](x, self.hidden)
        # out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]

        out = self.l1(x)
        return out

class FeatureAugmentation(nn.Module):
    def __init__(self, v_size, v_vec, dropout_ratio, n_layers, emb_dim=200, n_labels=2, gpu=True, batch_first=True):
        super(FeatureAugmentation, self).__init__()
        self.gpu = gpu
        self.h_dim = (emb_dim+34)//2
        self.word_embed = nn.Embedding(v_size, emb_dim, padding_idx=0)
        v_vec = torch.tensor(v_vec)
        self.word_embed.weight.data.copy_(v_vec)
        self.dropout_ratio = dropout_ratio
        self.n_layers = n_layers

        feature_embed_layers = []
        feature_embed_size = {
            "feature:0" : 25,
            "feature:1" : 26,
            "feature:2" : 12,
            "feature:3" : 6,
            "feature:4" : 94,
            "feature:5" : 32
        }
        for key in feature_embed_size:
            size = feature_embed_size[key]
            feature_embed = nn.Embedding(size, 5, padding_idx=0)
            feature_embed.weight.data[0] = torch.zeros(5)
            feature_embed_layers.append(feature_embed)
        self.feature_embed_layers = nn.ModuleList(feature_embed_layers)
        self.drop_target = nn.Dropout(p=dropout_ratio)

        common_lstm_layers = []
        for i in range(self.n_layers):
            lstm = nn.LSTM(input_size=emb_dim+34, hidden_size=self.h_dim, batch_first=batch_first, bidirectional=True)
            common_lstm_layers.append(lstm)
        self.common_lstm_layers = nn.ModuleList(common_lstm_layers)

        specific_lstm_layers = []
        for i in range(self.n_layers):
            specific_lstm = {}
            for domain in ['OC', 'OY', 'OW', 'PB', 'PM', 'PN']:
                lstm = nn.LSTM(input_size=emb_dim+34, hidden_size=self.h_dim, batch_first=batch_first, bidirectional=True)
                specific_lstm[domain] = lstm
            specific_lstm_layers.append(nn.ModuleDict(specific_lstm))
        self.specific_lstm_layers = nn.ModuleList(specific_lstm_layers)

        l1_layer = {}
        for domain in ['OC', 'OY', 'OW', 'PB', 'PM', 'PN']:
            l1 = nn.Linear(self.h_dim*2*2, n_labels)
            l1_layer[domain] = l1
        self.l1_layer = nn.ModuleDict(l1_layer)

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, x, domain):
        self.hidden = self.init_hidden(x[2].size(0))
        word_emb = self.word_embed(x[0])
        feature_emb_list = []
        for i, _x in enumerate(x[1]):
            feature_emb = self.feature_embed_layers[i](_x)
            feature_emb_list.append(feature_emb)
        x_feature = torch.tensor(x[2], dtype=torch.float, device=x[2].device)

        x = torch.cat(
            (word_emb, feature_emb_list[0], feature_emb_list[1], feature_emb_list[2], feature_emb_list[3], feature_emb_list[4], feature_emb_list[5], x_feature),
            dim=2
        )

        x = self.drop_target(x)

        out1 = x
        out2 = x
        for i in range(self.n_layers):
            out1, hidden = self.common_lstm_layers[i](out1, self.hidden)
            out2, hidden = self.specific_lstm_layers[i][domain](out2, self.hidden)

        out = torch.cat(
            (out1, out2), dim=2
        )
        out = self.l1_layer[domain](out)
        return out

class ClassProbabilityShift(nn.Module):
    def __init__(self, v_size, v_vec, dropout_ratio, n_layers, statistics_of_each_case_type, emb_dim=200, n_labels=2, gpu=True, batch_first=True):
        super(ClassProbabilityShift, self).__init__()
        self.gpu = gpu
        self.h_dim = (emb_dim+34)//2
        self.word_embed = nn.Embedding(v_size, emb_dim, padding_idx=0)
        v_vec = torch.tensor(v_vec)
        self.word_embed.weight.data.copy_(v_vec)
        self.dropout_ratio = dropout_ratio
        self.n_layers = n_layers

        self.init_statistics(statistics_of_each_case_type)

        feature_embed_layers = []
        feature_embed_size = {
            "feature:0" : 25,
            "feature:1" : 26,
            "feature:2" : 12,
            "feature:3" : 6,
            "feature:4" : 94,
            "feature:5" : 32
        }
        for key in feature_embed_size:
            size = feature_embed_size[key]
            feature_embed = nn.Embedding(size, 5, padding_idx=0)
            feature_embed.weight.data[0] = torch.zeros(5)
            feature_embed_layers.append(feature_embed)
        self.feature_embed_layers = nn.ModuleList(feature_embed_layers)
        self.drop_target = nn.Dropout(p=dropout_ratio)

        lstm_layers = []
        for i in range(self.n_layers):
            lstm = nn.LSTM(input_size=emb_dim+34, hidden_size=self.h_dim, batch_first=batch_first, bidirectional=True)
            lstm_layers.append(lstm)
        self.lstm_layers = nn.ModuleList(lstm_layers)
        self.l1 = nn.Linear(self.h_dim*2, n_labels)

    def init_statistics(self, statistics_of_each_case_type):
        max_length = 500
        media = statistics_of_each_case_type.keys()
        statistics_positive = {}
        for domain in media:
            intra = statistics_of_each_case_type[domain]['intra(dep)'] + statistics_of_each_case_type[domain]['intra(zero)']
            tmp = np.identity(max_length, dtype=np.float32)*intra
            for i, case_type in enumerate(['none', 'exoX', 'exo2', 'exo1']):
                tmp[i][i] = statistics_of_each_case_type[domain][case_type]
            statistics_positive[domain] = tmp
        all_I = np.matrix(statistics_positive['All']).I

        statistics_negative = {}
        for domain in media:
            statistics_positive[domain] *= all_I
            tmp = np.identity(max_length, dtype=np.float32) - statistics_positive[domain]
            statistics_negative[domain] = tmp

        for domain in media:
            statistics_positive[domain] = torch.tensor(statistics_positive[domain])
            statistics_negative[domain] = torch.tensor(statistics_negative[domain])
            if self.gpu:
                statistics_positive[domain] = statistics_positive[domain].cuda()
                statistics_negative[domain] = statistics_negative[domain].cuda()

        self.statistics_positive = statistics_positive
        self.statistics_negative = statistics_negative

    def CPS_layer(self, x, domains):
        sentence_length = x.size(1)
        out = x.clone()
        for i, domain in enumerate(domains):
            out[i] = torch.stack(
                [x[i, :, 0] * self.statistics_positive[domain][:sentence_length, :sentence_length].diag(),
                x[i, :, 1] * self.statistics_negative[domain][:sentence_length, :sentence_length].diag()
                ], dim=1
            )
        return out

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, x, domains):
        self.hidden = self.init_hidden(x[2].size(0))
        word_emb = self.word_embed(x[0])
        feature_emb_list = []
        for i, _x in enumerate(x[1]):
            feature_emb = self.feature_embed_layers[i](_x)
            feature_emb_list.append(feature_emb)
        x_feature = torch.tensor(x[2], dtype=torch.float, device=x[2].device)

        x = torch.cat(
            (word_emb, feature_emb_list[0], feature_emb_list[1], feature_emb_list[2], feature_emb_list[3], feature_emb_list[4], feature_emb_list[5], x_feature),
            dim=2
        )

        x = self.drop_target(x)
        for i in range(self.n_layers):
            x, hidden = self.lstm_layers[i](x, self.hidden)

        out = self.l1(x)
        out = self.CPS_layer(out, domains)
        return out
