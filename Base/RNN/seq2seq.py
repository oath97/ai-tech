'''
编码器将长度可变的输入序列转换成形状固定的上下文变量 c, 并将输入序列的信息在 c 中进行编码.
输入序列(b=1): x_1, x_2, ..., x_T
在时间步t, rnn 将词元 x_t 的输入特征向量和上一时间步的隐状态 h_t-1 转换为当前步的隐状态 h_t:
    h_t = f(x_t, h_t-1)
将所有时间步的隐状态转换为上下文变量:
    c = q(h_1, ..., h_T)
可以选择 q(h_1, ..., h_T) = h_T

编码器输出的上下文变量 c 对整个输入序列 {x_i} 进行编码; 
来自训练数据集的输出序列 {y_i} 对每个时间步 t' (与输入序列/编码器的时间步 t 不同), 解码器输出 y_t'
的概率取决于 y_i, ..., y_t'-1 和 c, 即 P(y_t'| y_1, ..., y_t'-1, c)
为了在序列上模型化条件概率, 可以使用另一个 rnn 作为解码器: s_t' = g(y_t'-1, c, s_t'-1)
获取解码器的隐状态后, 可以使用输出层和 softmax 操作, 来计算在 t' 时输出 y_t' 的条件分布概率.

实现解码器时, 可直接使用编码器最后一个时间步的隐状态来初始化解码器的隐状态, 这要求编解码器使用的 rnn 具有相同数量的层和隐藏单元.
为进一步包含经过编码的输入序列的信息, c 在所有时间步与解码器的输入进行 concatenate.
为了预测输出词元的概率分布, 在循环神经网络解码器的最后一层使用全连接层来变换隐状态.
'''

import collections
import math
import torch
from torch import nn

from Models.XFormer.transformer.base import Encoder, Decoder
from rnn import grad_clipping


class Seq2SeqEncoder(Encoder):
    '''序列到序列学习的循环神经网络编码器'''
    def __init__(self, 
                 vocab_size,
                 embed_size,
                 num_hiddens,
                 num_layers,
                 dropout=0,
                 **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
    
    def forward(self, X, *args):
        '''
        args:
            X:        (batch_size, num_steps)
        return:
            output    (num_steps, batch_size, num_hiddens)
            state     (num_layers, batch_size, num_hiddens)
        '''
        # 获取每个词元的特征向量
        X = self.embedding(X)           # X         (batch_size, num_steps, embed_size)
        
        X = X.permute(1, 0, 2)          # X         (num_steps, batch_size, embed_size)
        
        output, state = self.rnn(X)     # output    (num_steps, batch_size, num_hiddens)
                                        # state     (num_layers, batch_size, num_hiddens)
        return output, state
    
    
class Seq2SeqDecoder(Decoder):
    '''序列到序列学习的循环神经网络解码器'''
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    
    def forward(self, X, state):
        '''
        args:
            X:        (batch_size, num_steps)
            state     (num_layers, batch_size, num_hiddens)
        return:
            output:   (batch_size, num_steps, num_hiddens)
state       state:    (num_layers, batch_size, num_hiddens)
        '''
        X = self.embedding(X)           # X         (batch_size, num_steps, embed_size)
        X = X.permute(1, 0, 2)          # X         (num_steps, batch_size, embed_size)
        
        # 广播 context, 使之具有与 X 相同的 num_steps
        context = state[-1]                         # context   (batch_size, num_hiddens)
        context = context.repeat(X.shape[0], 1, 1)  # context   (num_steps, batch_size, num_hiddens)
        
        X_and_context = torch.cat((X, context), 2)  # X_and_context (num_steps, batch_size, embed_size + num_hiddens)
        
        output, state = self.rnn(X_and_context, state)  # output    (num_steps, batch_size, num_hiddens)
                                                        # state     (num_layers, batch_size, num_hiddens)
        
        output = self.dense(output)                     # output    (num_steps, batch_size, vocab_size)
        output = output.permute(1, 0, 2)                # output    (batch_size, num_steps, vocab_size)
        
        return output, state
    
    
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    '''带掩蔽到 softmax 交叉熵损失函数'''
    def forward(self, pred, label, valid_len):
        '''
        args:
            pred:       (batch_size, num_steps, vocab_size)
            label:      (batch_size, num_steps)
            valid_len:  (batch_size,)
        return:
            weighted_loss (batch_size,)
        '''
        weights = torch.ones_like(label)                # weights (batch_size, num_steps)
        weights = sequence_mask(weights, valid_len)     # weights (batch_size, num_steps)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)    # unweighted_loss   (batch_size, num_steps)
        weighted_loss = (unweighted_loss * weights).mean(dim=1) # weighted_loss (batch_size,)
        return weighted_loss


def sequence_mask(X, valid_len, value=0):
    '''在序列中屏蔽不相关的项
    args:
        X:          (batch_size, num_steps)
        valid_len:  (batch_size,)
    return:
        X:          (batch_size, num_steps)
    '''
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,                  # [None, :] (n,)-->(1, n) 
                        device=X.device)[None, :] < valid_len[:, None]  # [:, None] (n,)-->(n, 1)
    X[~mask] = value
    return X


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    '''训练序列到序列模型'''
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    
    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            
            dec_input = torch.cat([bos, Y[:, :-1]], 1) # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            
            grad_clipping(net, 1)
            
            optimizer.step()