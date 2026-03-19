import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer  
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

import logging
logger = logging.getLogger(__name__)

class Transformer(nn.Module):
    def __init__(self, enc_in, c_out, seq_len, label_len, out_len, dec_in,  
                factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Transformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)  
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        logger.info(f"Attention type is: {Attn.__name__}") 
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        

        self.projection_conv = nn.ConvTranspose1d(in_channels=d_model, out_channels=c_out, kernel_size=4,  # Kernel size chosen to facilitate doubling the length
                                                          stride=2,       # Stride of 2 upsamples by a factor of 2
                                                          padding=1,       # Padding to maintain correct output size
                                                          output_padding=1)    #Lout=(Lin −1)×stride−2×padding+kernel_size+output_padding
        
        
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        logger.debug(f"Input embeddding for Encoder")
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        logger.debug(f"emb_out:{enc_out.size()},{enc_out}") 
        
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        logger.debug(f"enc_out= {enc_out.shape,enc_out},\n attns ={len(attns),attns[0].shape,attns}") 
        
        

        enc_out = self.projection_conv(enc_out.transpose(1,2))  
        logger.debug(f"enc_out_proj:{enc_out.size()},{enc_out}")

        
        dec_out =x_dec#self.projection(dec_out)  
        logger.debug(f"enc_out_proj:{dec_out.size()},{dec_out}") 
        
        if self.output_attention:
            return [enc_out.transpose(1,2),dec_out], attns 
        else:
            return [enc_out.transpose(1,2),dec_out] 
