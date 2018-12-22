import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from allennlp.modules.elmo import Elmo, batch_to_ids

class BiLSTM(nn.Module):
    def __init__(self, emb_dim, n_labels, v_size, v_vec, dropout_ratio, n_layers, gpu=True, batch_first=True):
        super(BiLSTM, self).__init__()
        self.gpu = gpu
        self.h_dim = emb_dim+34
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
            x, hidden = self.lstm(x, self.hidden)
        # out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        
        out = self.l1(out)
        return out

class FeatureAugmentation(nn.Module):
    def __init__(self, emb_dim, h_dim, n_labels, v_size, gpu=True, v_vec=None, batch_first=True):
        super(FeatureAugmentation, self).__init__()
        self.gpu = gpu
        self.h_dim = h_dim
        self.word_embed = nn.Embedding(v_size, emb_dim, padding_idx=0)
        v_vec = torch.tensor(v_vec)
        self.word_embed.weight.data.copy_(v_vec)

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

        self.common_lstm = nn.LSTM(input_size=emb_dim+34, hidden_size=h_dim, batch_first=batch_first, bidirectional=True)

        specific_lstm_layers = {}
        for domain in ['OC', 'OY', 'OW', 'PB', 'PM', 'PN']:
            specific_lstm = nn.LSTM(input_size=emb_dim+34, hidden_size=h_dim, batch_first=batch_first, bidirectional=True)
            specific_lstm_layers[domain] = specific_lstm
        self.specific_lstm_layers = nn.ModuleDict(specific_lstm_layers)

        self.l1 = nn.Linear(h_dim*2*2, n_labels)

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

        out1, hidden = self.common_lstm(x, self.hidden)
        # out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]

        out2, hidden = self.specific_lstm_layers[domain](x, self.hidden)
        # TODO: concat!!!
        out = self.l1(out)
        return out


# class ClassProbabilityShift(nn.Module):
#     def __init__(self, emb_dim, h_dim, n_labels, v_size, gpu=True, v_vec=None, batch_first=True, type_statistics):
#         super(ClassProbabilityShift, self).__init__()
#         self.gpu = gpu
#         self.h_dim = h_dim
#         self.word_embed = nn.Embedding(v_size, emb_dim, padding_idx=0)
#         v_vec = torch.tensor(v_vec)
#         self.word_embed.weight.data.copy_(v_vec)

#         feature_embed_layers = []
#         feature_embed_size = {
#             "feature:0" : 25,
#             "feature:1" : 26,
#             "feature:2" : 12,
#             "feature:3" : 6,
#             "feature:4" : 94,
#             "feature:5" : 32
#         }
#         for key in feature_embed_size:
#             size = feature_embed_size[key]
#             feature_embed = nn.Embedding(size, 5, padding_idx=0)
#             feature_embed.weight.data[0] = torch.zeros(5)
#             feature_embed_layers.append(feature_embed)
#         self.feature_embed_layers = nn.ModuleList(feature_embed_layers)

#         self.lstm = nn.LSTM(input_size=emb_dim+34, hidden_size=h_dim, batch_first=batch_first, bidirectional=True)
#         self.l1 = nn.Linear(h_dim*2, n_labels)

#     def _calculate_type_statistics(self):
#         domain_statistics_positive = {}

#         tmp = np.full((sentence_length, ), type_statistics_dict['All'][-1])
#         for i in range(4):
#             tmp[i] = type_statistics_dict['union'][i]
#         tmp = np.diag(tmp)
#         union_I = np.matrix(tmp, dtype=np.float32).I

#         for domain in type_statistics_dict.keys():
#             if domain == 'union':
#                 continue
#             statistics = type_statistics_dict[domain]
#             tmp = np.full((sentence_length, ), statistics[-1])
#             for i in range(0, 4):
#                 tmp[i] = statistics[i]
#             tmp = np.diag(tmp)
#             domain_statistics_positive[domain] = np.matrix(tmp, dtype=np.float32)*union_I
#             domain_statistics_positive[domain] = chainer.dataset.to_device(device, domain_statistics_positive[domain])

#         for domain in type_statistics_dict.keys():
#             type_statistics_dict[domain] = [100-type_statistics_dict[domain][i] for i in range(5)]

#         domain_statistics_negative = {}

#         tmp = np.full((sentence_length, ), type_statistics_dict['union'][-1])
#         for i in range(4):
#             tmp[i] = type_statistics_dict['union'][i]
#         tmp = np.diag(tmp)
#         union_neg_I = np.matrix(tmp, dtype=np.float32).I

#         for domain in type_statistics_dict.keys():
#             if domain == 'union':
#                 continue
#             statistics = type_statistics_dict[domain]
#             tmp = np.full((sentence_length, ), statistics[-1])
#             for i in range(0, 4):
#                 tmp[i] = statistics[i]
#             tmp = np.diag(tmp)
#             domain_statistics_negative[domain] = np.matrix(tmp, dtype=np.float32)*union_neg_I
#             domain_statistics_negative[domain] = chainer.dataset.to_device(device, domain_statistics_negative[domain])

#     def init_hidden(self, b_size):
#         h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
#         c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
#         if self.gpu:
#             h0 = h0.cuda()
#             c0 = c0.cuda()
#         return (h0, c0)

#     def forward(self, x):
#         self.hidden = self.init_hidden(x[2].size(0))
#         word_emb = self.word_embed(x[0])
#         feature_emb_list = []
#         for i, _x in enumerate(x[1]):
#             feature_emb = self.feature_embed_layers[i](_x)
#             feature_emb_list.append(feature_emb)
#         x_feature = torch.tensor(x[2], dtype=torch.float, device=x[2].device)

#         x = torch.cat(
#             (word_emb, feature_emb_list[0], feature_emb_list[1], feature_emb_list[2], feature_emb_list[3], feature_emb_list[4], feature_emb_list[5], x_feature),
#             dim=2
#         )

#         out, hidden = self.lstm(x, self.hidden)
#         # out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        
#         out = self.l1(out)
#         return out