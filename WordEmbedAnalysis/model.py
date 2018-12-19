import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from allennlp.modules.elmo import Elmo, batch_to_ids

class BiLSTM(nn.Module):
    def __init__(self, emb_dim, h_dim, n_labels, v_size, gpu=True, v_vec=None, batch_first=True, emb_type=None, elmo_model_dir=None):
        super(BiLSTM, self).__init__()
        self.gpu = gpu
        self.h_dim = h_dim
        if emb_type=='ELMo':
            options_file = f'{elmo_model_dir}/options.json'
            weight_file = f'{elmo_model_dir}/weights.hdf5'
            self.word_embed = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
            if gpu:
                self.word_embed = self.word_embed.cuda()
        elif emb_type=='ELMoForManyLangs':
            from elmoformanylangs import Embedder
            e = Embedder(elmo_model_dir)
            self.word_embed = e.sents2elmo
        else:
            self.word_embed = nn.Embedding(v_size, emb_dim, padding_idx=0)
        if v_vec is not None:
            v_vec = torch.tensor(v_vec)
            self.word_embed.weight.data.copy_(v_vec)

        feature_embed_layers = []
        feature_embed_size = {"feature:0":25, "feature:1":26, "feature:2":12, "feature:3":6, "feature:4":94, "feature:5":32}
        for key in feature_embed_size:
            size = feature_embed_size[key]
            feature_embed = nn.Embedding(size, 5, padding_idx=0)
            feature_embed.weight.data[0] = torch.zeros(5)
            feature_embed_layers.append(feature_embed)
        self.feature_embed_layers = nn.ModuleList(feature_embed_layers)

        self.lstm = nn.LSTM(input_size=emb_dim+36, hidden_size=h_dim, batch_first=batch_first, bidirectional=True)
        self.l1 = nn.Linear(h_dim*2, n_labels)

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
        if self.word_embed.__class__.__name__ == 'Embedding':
            pass
        elif self.word_embed.__class__.__name__ == 'Elmo':
            exophoras = [['これ'], ['あなた'], ['私']]
            exophora_ids = batch_to_ids(exophoras)
            if self.gpu:
                exophora_ids = exophora_ids.cuda()
            exophora_emb = self.word_embed(exophora_ids)
            word_emb = word_emb['elmo_representations'][0]
            exophora_emb = exophora_emb['elmo_representations'][0]
            exophora_emb = exophora_emb.reshape(3, -1)
            exophora_emb = exophora_emb.repeat([word_emb.shape[0], 1, 1])
            none_emb = torch.zeros(word_emb.shape[0], 1, word_emb.shape[2])
            if self.gpu:
                none_emb = none_emb.cuda()
            word_emb = torch.cat((none_emb, exophora_emb, word_emb), 1)
        elif self.word_embed.__func__.__name__ == 'sents2elmo':
            word_emb = [torch.tensor(emb) for emb in word_emb]
            word_emb = nn.utils.rnn.pad_sequence(word_emb, batch_first=True, padding_value=0)
            exophoras = [['これ'], ['あなた'], ['私']]
            exophora_emb = self.word_embed(exophoras)
            exophora_emb = torch.tensor(exophora_emb).reshape(3, -1)
            exophora_emb = exophora_emb.repeat([word_emb.shape[0], 1, 1])
            none_emb = torch.zeros(word_emb.shape[0], 1, word_emb.shape[2])
            if self.gpu:
                word_emb = word_emb.cuda()
                exophora_emb = exophora_emb.cuda()
                none_emb = none_emb.cuda()
            word_emb = torch.cat((none_emb, exophora_emb, word_emb), 1)
        feature_emb_list = []
        for i, _x in enumerate(x[1]):
            feature_emb = self.feature_embed_layers[i](_x)
            feature_emb_list.append(feature_emb)
        x_feature = torch.tensor(x[2], dtype=torch.float, device=x[2].device)
        x = torch.cat((word_emb, feature_emb_list[0], feature_emb_list[1], feature_emb_list[2], feature_emb_list[3], feature_emb_list[4], feature_emb_list[5], x_feature), 2)
        out, hidden = self.lstm(x, self.hidden)
        # out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        
        out = self.l1(out)
        return out
