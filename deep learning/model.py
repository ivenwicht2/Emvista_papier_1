import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np

class LSTM(nn.Module):
    def __init__(self,len_vocab,len_output, hidden_dim, emb_dim=300,
                 spatial_dropout=0.05, recurrent_dropout=0.1, num_linear=1):
        super().__init__() 
        self.embedding = nn.Embedding(len_vocab, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1, dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, len_output)
    
    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds


class Camembert(nn.Module):
    def __init__(self,len_vocab,len_output, hidden_dim, emb_dim=300, recurrent_dropout=0.1):
        super().__init__() 
        self.bert = BertModel.from_pretrained("camembert-base")
        self.drop = nn.Dropout(p=0.3)
        self.predictor = nn.Linear(self.bert.config.hidden_size, len_output)

    def forward(self, seq):
        output, feature = self.bert(input_ids=seq)
        output = output[-1, :, :]
        feature = self.drop(output)
        preds = self.predictor(feature)
        return preds