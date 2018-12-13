import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BiLSTM(nn.Module):
    def __init__(self, emb_dim, h_dim, n_labels, v_size, gpu=True, v_vec=None, batch_first=True):
        super(BiLSTM, self).__init__()
        self.gpu = gpu
        self.h_dim = h_dim
        self.word_embed = nn.Embedding(v_size, emb_dim)
        if v_vec is not None:
            self.word_embed.weight.data.copy_(v_vec)
        feature_embed_size = {"feature:0":24, "feature:1":25, "feature:2":11, "feature:3":5, "feature:4":93, "feature:5":31}
        self.feature_embed_list = []
        for i, key in enumerate(feature_embed_size):
            size = feature_embed_size[key]
            embed = nn.Embedding(size, 5)
            embed.weight.data[0] = torch.zeros(5)
            self.feature_embed_list.append(embed)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=h_dim, batch_first=batch_first, bidirectional=True)
        self.l1 = nn.Linear(h_dim*2, n_labels)

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.hdim))
        if self.gpu:
            h0 = h0.cuda()
            co = c0.cuda()
        return (h0, c0)

    def forward(self, sentence, lengths=None):
        self.hidden = self.init_hidden(sentence.size(0))
        emb = self.word_embed(sentence)
        out, hidden = self.lstm(emb, self.hidden)

        out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        
        out = self.l1(out)

        return out
