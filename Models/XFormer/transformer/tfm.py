import math
from torch import nn
from base import Encoder, AttentionDecoder
from blk import EncoderBlock, DecoderBlock
from pe import PositionalEncoding


class TransformerEncoder(Encoder):
    '''Transformer 编码器'''
    def __init__(self, vocab_size, key_size, query_size, value_size, 
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block'+str(i),
                                 EncoderBlock(key_size, query_size, value_size, 
                                              num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                                              num_heads, dropout, use_bias))
        
    def forward(self, X, valid_lens, *args):
        '''
        args:
            X:          tensor (batch_size, seq_len)
            valid_lens: tensor (batch_size,) or (batch_size, seq_len)
        return:
            X:          tensor (batch_size, seq_len, d_model)
        '''
        # 因为位置编码在[-1,1]，因此嵌入值乘以嵌入维度的平方根进行缩放，再与位置编码相加
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
    

class TransformerDecoder(AttentionDecoder):
    '''Transformer 解码器'''
    def __init__(self, vocab_size, key_size, query_size, value_size, 
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block'+str(i),
                                 DecoderBlock(key_size, query_size, value_size, 
                                              num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                                              num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self, enc_outputs, enc_valid_len, *args):
        return [enc_outputs, enc_valid_len, [None] * self.num_layers]
    
    def forward(self, X, state):
        '''
        args:
            X:          tensor (batch_size, seq_len)
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
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # 编码器-解码器自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    
    @property
    def attention_weights(self):
        return self._attention_weights
    
