import torch
from torch import nn


class AddNorm(nn.Module):
    '''残差连接 + 层归一化'''
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
    
    
if __name__ == '__main__':
    
    add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    X, Y = torch.arange(2*3*4*2, dtype=torch.float32).reshape(-1, 2, 3, 4)
    Z = add_norm(X, Y)
    
    print(f"AddNorm: \n \
        X = {X}  \n \
        Y = {Y}  \n \
        Z = {Z}")