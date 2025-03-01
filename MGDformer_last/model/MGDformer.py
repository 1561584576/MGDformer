import torch
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.filter import UNet
from layers.Embed import DataEmbedding_inverted,Dimission_Embedding
from model.revin import RevIN
import torch.nn as nn

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.individual = configs.individual
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    UNet(configs),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
                # ) for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,y_enc):
        # print(x_dec.shape)
        # print("====================")


        if self.use_norm:

            rev = RevIN(x_enc.shape[2])

            x_enc = rev(x_enc, 'norm')


        batch, _, N = x_enc.shape # B L N


        if self.individual:
            x_enc = self.enc_embedding(x_enc)
            enc_out= self.encoder(x_enc, attn_mask=None)
            # B N E -> B N S -> B S N
            dec_out = self.projector(enc_out).permute(0, 2, 1)  # filter the covariates


        if self.use_norm:
            dec_out = rev(dec_out, 'denorm')


        return dec_out,y_enc


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,y_enc, mask=None):
        dec_out,y_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,y_enc)
        return dec_out[:, -self.pred_len:, :],y_out[:, -self.pred_len:, :]  # [B, L, D]