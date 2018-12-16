import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BiLSTM(nn.Module):
    def __init__(self, emb_dim, h_dim, n_labels, v_size, gpu=True, v_vec=None, batch_first=True):
        super(BiLSTM, self).__init__()
        self.gpu = gpu
        self.h_dim = h_dim
        self.word_embed = nn.Embedding(v_size, emb_dim, padding_idx=0)
        if v_vec is not None:
            v_vec = torch.tensor(v_vec)
            self.word_embed.weight.data.copy_(v_vec)

        feature_embed_size = {"feature:0":25, "feature:1":26, "feature:2":12, "feature:3":6, "feature:4":94, "feature:5":32}
        size = feature_embed_size["feature:0"]
        self.feature0_embed = nn.Embedding(size, 5, padding_idx=0)
        self.feature0_embed.weight.data[0] = torch.zeros(5)
        size = feature_embed_size["feature:1"]
        self.feature1_embed = nn.Embedding(size, 5, padding_idx=0)
        self.feature1_embed.weight.data[0] = torch.zeros(5)
        size = feature_embed_size["feature:2"]
        self.feature2_embed = nn.Embedding(size, 5, padding_idx=0)
        self.feature2_embed.weight.data[0] = torch.zeros(5)
        size = feature_embed_size["feature:3"]
        self.feature3_embed = nn.Embedding(size, 5, padding_idx=0)
        self.feature3_embed.weight.data[0] = torch.zeros(5)
        size = feature_embed_size["feature:4"]
        self.feature4_embed = nn.Embedding(size, 5, padding_idx=0)
        self.feature4_embed.weight.data[0] = torch.zeros(5)
        size = feature_embed_size["feature:5"]
        self.feature5_embed = nn.Embedding(size, 5, padding_idx=0)
        self.feature5_embed.weight.data[0] = torch.zeros(5)
        self.lstm = nn.LSTM(input_size=emb_dim+35, hidden_size=h_dim, batch_first=batch_first, bidirectional=True)
        self.l1 = nn.Linear(h_dim*2, n_labels)

        self.softmax = nn.Softmax(dim=2)

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, x_wordemb, x_featureemb0, x_featureemb1, x_featureemb2, x_featureemb3, x_featureemb4, x_featureemb5, x_feature):
        self.hidden = self.init_hidden(x_wordemb.size(0))
        word_emb = self.word_embed(x_wordemb)
        feature0_emb = self.feature0_embed(x_featureemb0)
        feature1_emb = self.feature1_embed(x_featureemb1)
        feature2_emb = self.feature2_embed(x_featureemb2)
        feature3_emb = self.feature3_embed(x_featureemb3)
        feature4_emb = self.feature4_embed(x_featureemb4)
        feature5_emb = self.feature5_embed(x_featureemb5)
        x_feature = torch.tensor(x_feature, dtype=torch.float, device=x_feature.device)
        x = torch.cat((word_emb, feature0_emb, feature1_emb, feature2_emb, feature3_emb, feature4_emb, feature5_emb, x_feature), 2)
        out, hidden = self.lstm(x, self.hidden)
        # out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        
        out = self.l1(out)
        # out = self.softmax(out)

        return out
