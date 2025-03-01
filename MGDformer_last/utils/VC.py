import torch
import torch.nn as nn
from layers.VCformer_Enc import Encoder, EncoderLayer
from layers.SelfAttention_Family import VarCorAttention, VarCorAttentionLayer


class VCformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2405.11470
    """

    def __init__(self, configs):
        super(VCformer, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in=configs.enc_in

        # Encoder

        self.encoder = Encoder(
            [
                EncoderLayer(
                    VarCorAttentionLayer(
                        VarCorAttention(configs, False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(3)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder

        self.projection = nn.Linear(
            configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):

        enc_out, attns = self.encoder(x_enc, attn_mask=None)

        return enc_out, attns

    def forward(self, x_enc):

        dec_out, attns = self.forecast(
            x_enc)
        # return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return dec_out

