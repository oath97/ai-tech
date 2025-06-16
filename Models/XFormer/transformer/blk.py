import torch
from torch import nn
from attention import MultiHeadAttention
from addnorm import AddNorm
from ffn import PositionWiseFNN


class EncoderBlock(nn.Module):
    '''Transformer 编码器块'''
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, 
                                            num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFNN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_lens):
        '''
        args:
            X:          tensor (batch_size, seq_len, d_model)
            valid_lens: tensor (batch_size,) or (batch_size, seq_len)
        return:
            tensor (batch_size, seq_len, d_model)
        '''
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
    
    
class DecoderBlock(nn.Module):
    '''Transformer 解码器块'''
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, 
                                            num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, 
                                            num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFNN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
        
    
    def forward(self, X, state):
        '''
        args:
            X:          tensor (batch_size, seq_len, d_model)
            state:      list
                            0:enc_outputs:  tensor (batch_size, seq_len, d_model)
                            1:valid_lens:   tensor (batch_size,) or (batch_size, seq_len)
                            2:dec_outputs:  [tensor] (N, seq_len, d_model) or [None]
        return:
            X:          tensor (batch_size, seq_len, d_model)
            state:      list
                            0:enc_outputs:  tensor (batch_size, seq_len, d_model)
                            1:valid_lens:   tensor (batch_size,) or (batch_size, seq_len)
                            2:dec_outputs:  [tensor] (N, seq_len, d_model) or [None]
        '''
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，因此state[2][self.i]初始化为None
        # 预测阶段，输出序列是通过词元一个接着一个解码的，因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat(([state[2][self.i], X]), axis=1)
        state[2][self.i] = key_values
        
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_len = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_len = None
        
        X2 = self.attention1(X, key_values, key_values, dec_valid_len)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
        