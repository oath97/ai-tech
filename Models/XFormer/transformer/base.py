from torch import nn


class Encoder(nn.Module):
    '''base encoder'''
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
    
    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    '''base encoder'''
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
        
    def forward(self, X, *args):
        raise NotImplementedError
    
    
class AttentionDecoder(Decoder):
    '''base attention-based decoder'''
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
    
    @property
    def attetion_weights(self):
        raise NotImplementedError