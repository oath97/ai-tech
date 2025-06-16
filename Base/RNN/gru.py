'''
RNN 中计算梯度,矩阵连续成绩可能导致梯度消失/爆炸:
    1.早期观测值对预测所有未来观测值具有非常重要的意义:
        希望有某些机制能够在一个记忆元里存储重要的早期信息。 如果没有这样的机制,
        我们将不得不给这个观测值指定一个非常大的梯度, 因为它会影响所有后续的观测值。
    2.一些词元可能没有相关的观测值:
        希望有一些机制来跳过隐状态表示中的此类词元。
    3.序列的各个部分之间存在逻辑中断:
        最好有一种方法来重置我们的内部状态表示。
        
LSTM 和 GRU 都是解决这类问题的方法。


GRU 和普通 RNN 之间的关键区别在于: GRU 支持隐状态的门控，即模型有专门的机制（可学习）来确定应合适更新/重制隐状态。
    重制门：允许控制“可能还想记住”的过去状态的数量
        R_t = sigmoid(X_t * W_xr + H_t-1 * W_hr + b_r) in (0,1)
        X_t:    (b, d)
        H_t-1:  (b, h)
    更新门：允许控制新状态中有多少个是旧状态的副本
        Z_t = sigmoid(X_t * W_xz + H_t-1 * W_hz + b_z) in (0,1)
    候选隐状态: R_t->1 普通 RNN; R_t->0 MLP(X_t), 预先存在的任何隐状态会被重置为默认值
        H^t = tanh(X_t * W_xh + (R_t·H_t-1) * W_hh + b_h) in (-1,1)
    隐状态: 多大程度上来自旧的状态 H_t-1 和新的候选状态 H^t; Z_t->0, H_t->H^t
        H_t = Z_t·H_t-1 + (1 - Z_t)·H^t
'''

import torch


def get_params(vocab_size, num_hiddens, device):
    '''初始化模型参数: 从标准差为0.01的高斯分布中提取参数, 并将偏置项设为0'''
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    
    W_xz, W_hz, b_z = three()   # 更新门参数
    W_xr, W_hr, b_r = three()   # 重置门参数
    W_xh, W_hh, b_h = three()   # 候选隐状态参数
    
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    '''初始化隐状态 (b,h)'''
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def gru(inputs, state, params):
    '''门控循环单元模型
    args:
        inputs: (num_steps, batch_size, input_dim)
        state:  (batch_size, num_hiddens)
    output:
        inputs: (num_steps, batch_size, num_hiddens)
        state:  (batch_size, num_hiddens)
    '''
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

    