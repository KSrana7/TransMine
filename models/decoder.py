import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.linear1 = nn.Linear(d_model, d_ff) 
        self.linear2 = nn.Linear(d_ff, d_model) 


    def forward(self, x, cross, x_mask=None, cross_mask=None):
        
        logger.debug(f"CACLULATING SELF ATTENTION IN DECODER:") 
        
        new_x, self_attn = self.self_attention(
            x, x, x,
            attn_mask=x_mask
            )
        x = x + self.dropout(new_x)
        
        # x = x + self.dropout(self.self_attention(
        #     x, x, x,
        #     attn_mask=x_mask
        # )[0])
        x = self.norm1(x)
        
        logger.debug(f"CACLULATING CROSS ATTENTION IN DECODER:") 
        
        new_x, cross_attn = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
            )
        
        x = x + self.dropout(new_x)

        
        #---------Transformer style FFN layer----------------------->
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        y_out = self.norm3(x + y)
        #---------------------------------------------------------------->


        return y_out, self_attn, cross_attn   

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        self_attns = []                 
        cross_attns = []                
        dlayer = 0
        
        for layer in self.layers:
            dlayer=dlayer+1
            logger.debug(f"Decoder layer:{dlayer}")

            x,self_attn,cross_attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)  
            
            self_attns.append(self_attn)   
            cross_attns.append(cross_attn) 

        if self.norm is not None:
            x = self.norm(x)

        return x,self_attns,cross_attns  