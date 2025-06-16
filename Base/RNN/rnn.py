'''
循环神经网络, 是具有隐状态的神经网络.

单隐藏层的MLP: 
    H = phi(X * W_xh + b_h)
    O = H * W_hq + b_q

有隐藏状态的 rnn: 保存了前一个时间步的隐藏变量
    H_t = phi(X_t * W_xh + H_t-1 * W_hh + b_h)
    O_t = H_t * W_hq + b_q

困惑度: 度量语言模型的质量.
如果想要压缩文本, 可以根据当前词元集预测的下一个词元.
一个更好的语言模型应该更准确地预测下一个词元, 因此它应该允许在压缩序列时花费更少的比特.
可以通过一个序列中所有的 n 个词元的交叉熵损失的平均值来衡量:
   (- log[P(x_2 | x_1)] - ... - log[P(x_t | x_t-1, ..., x_1)]) / n
困惑度是上式的指数.
'''

import torch
from torch import nn


def grad_clipping(net, theta):
    '''裁剪梯度'''
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm