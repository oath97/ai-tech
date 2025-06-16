import math
import torch
from torch import nn

def sequence_mask(X, valid_len, value=0):
    '''在序列中掩蔽不相关的项
    args:
        X:          tensor (batch_size * seq_len, d_model)
        valid_len:  tensor (batch_size * seq_len,)
    return:
        tensor (batch_size * seq_len, d_model)
    '''
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    '''通过在最后一个轴上掩蔽元素来执行softmax操作
    args:
        X:          tensor (batch_size, seq_len, d_model)
        valid_lens: tensor (batch_size,) or (batch_size, seq_len)
    return:
        tensor (batch_size, seq_len, d_model)
    '''
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # d_model轴上被掩蔽的元素使用一个非常大的负值替换，从而使其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    '''缩放点积注意力'''
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, valid_lens=None):
        '''
        args:
            queries:    tensor (batch_size, seq_len_q, d_model)
            keys:       tensor (batch_size, seq_len, d_model)
            values:     tensor (batch_size, seq_len, d_v)
            valid_lens: tensor (batch_size,) or (batch_size, seq_len)
        return:
            tensor (batch_size, seq_len_q, d_v)
        '''
        d = queries.shape[-1]
        
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    

class MultiHeadAttention(nn.Module):
    '''多头注意力'''
    def __init__(self, key_size, query_size, value_size, d_model, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, d_model, bias=bias)
        self.W_k = nn.Linear(key_size, d_model, bias=bias)
        self.W_v = nn.Linear(value_size, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
    
    def forward(self, queries, keys, values, valid_lens):
        '''
        args:
            queries:    tensor (batch_size, seq_len_q, d_model)
            keys:       tensor (batch_size, seq_len, d_model)
            values:     tensor (batch_size, seq_len, d_v)
            valid_lens: tensor (batch_size,) or (batch_size, seq_len)
        return:
            tensor (batch_size, seq_len_q, d_v)
        '''
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_q(keys), self.num_heads)
        values = transpose_qkv(self.W_q(values), self.num_heads)
        
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        
        return self.W_o(output_concat)
        
        
def transpose_qkv(X:torch.Tensor, num_heads):
    '''为了多头注意力的并行计算而变换形状
    args:
        X: tensor (batch_size, seq_len, d_model)
    return:
        tensor (batch_size * num_heads, seq_len, d_model / num_heads)
    '''
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])
        

def transpose_output(X:torch.Tensor, num_heads):
    '''逆转transpose_qkv操作
    args:
        X: tensor (batch_size * num_heads, seq_len_q, d_v / num_heads)
    return:
        tensor (batch_size, seq_len_q, d_v)
    '''
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
