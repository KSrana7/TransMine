import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import logging
logger = logging.getLogger(__name__)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model): #-----------------------------------c_in is enc_in(# of input features)--------->
        super(TokenEmbedding, self).__init__()

        #------------------------------Series Feature Extraction----------->
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=d_model//2, 
                                    kernel_size=3, dilation=2,padding=2, padding_mode='reflect') 

        self.pool1 = nn.MaxPool1d(kernel_size=2) 
        
        self.conv2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model, 
                            kernel_size=3, dilation=1,padding=1, padding_mode='reflect') 
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

        #------------------------------------------------------------------>


    def forward(self, x):
        
        #------------------------------Series Feature Extraction----------->
        x1 = self.conv1(x.permute(0, 2, 1))
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        # x3 = torch.cat([x2,x3], dim=1)
        x = x3.transpose(1,2)
        #----------------------------------------------------------------->

        return x
    

class TokenEmbedding_lookup(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding_lookup, self).__init__()

      
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.embedding = nn.Linear(vocab_size, d_model)
        logger.debug(f"Token Embedding is : {self.embedding} ")

    def forward(self, x):
        #-------For nn.embedding----------------->
        x= x.int().long() 
        batch_size, seq_len, num_features = x.size()
        x = x.view(batch_size * seq_len, num_features)
        logger.debug('x_embed',x.size(),x)
        x_out = self.embedding(x)
        x_out = x_out.view(batch_size, seq_len, -1)
        
        return x_out

#-------------------------------------------------------------------------------------------------------------------------->

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, enc_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        vocab_size = enc_in 
        self.d_model = d_model


        self.value_embedding = TokenEmbedding(c_in=enc_in, d_model=d_model)        
        self.position_embedding = PositionalEmbedding(d_model=d_model)    
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        logger.debug('Temporal embedding not added')
        
        #------------Added Convolution for token embedding----------------->
        logger.debug('CNN for embedding')
        x_out = self.value_embedding(x)
        x = self.position_embedding(x_out) + x_out #----------Commented + self.temporal_embedding(x_mark) to not add temporal features to the model---->
        
        return self.dropout(x)
