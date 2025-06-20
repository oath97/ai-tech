import torch
from torch import nn


class PositionalEncoding(nn.Module):
    '''位置编码'''
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) 
        X = X / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        '''emb加入位置编码
        args:
            X: tensor (1, seq_len, d_model), d_model = num_hiddens
        return:
            tensor
        '''
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)