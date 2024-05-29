import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_wo_pos
from models.embed import DataEmbedding, DataEmbedding_inverted
from models.local_global import Seasonal_Prediction, series_decomp_multi


class MICN(nn.Module):
    def __init__(self, configs,dec_in, c_out, seq_len, label_len, out_len,
                 d_model=512, n_heads=8,d_layers=2,
                 dropout=0.0,embed='fixed', freq='h',
                 device=torch.device('cuda:0'), mode='regre',
                 decomp_kernel=[33], conv_kernel=[12, 24], isometric_kernel=[18, 6],):
        super(MICN, self).__init__()
        self.configs = configs
        self.pred_len = out_len
        self.seq_len = seq_len
        self.c_out = c_out
        self.decomp_kernel = configs.decomp_kernel
        self.mode = mode
        self.use_norm = configs.use_norm
        self.decomp_multi = series_decomp_multi(self.decomp_kernel)
        # embedding
        self.inverted_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.conv_trans = Seasonal_Prediction(configs=configs,embedding_size=d_model, n_heads=n_heads, dropout=dropout,
                                     d_layers=d_layers, decomp_kernel=self.decomp_kernel, c_out=c_out, conv_kernel=conv_kernel,
                                     isometric_kernel=isometric_kernel, device=device)

        self.regression = nn.Linear(seq_len, out_len)
        self.regression.weight = nn.Parameter((1/out_len) * torch.ones([out_len, seq_len]), requires_grad=True)

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # 2024.4.1新增 begin
        if self.configs.use_x_mark_enc == 1:
            x_mark_enc = x_mark_enc
        else:
            x_mark_enc = None

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N

        if self.configs.front_use_decomp == 1:
            seasonal_init_enc, trend = self.decomp_multi(x_enc)
            trend = self.regression(trend.permute(0,2,1)).permute(0, 2, 1)
        else:
            seasonal_init_enc = x_enc
            trend = None

        seasonal_init_dec = seasonal_init_enc

        dec_out = self.inverted_embedding(seasonal_init_dec,x_mark_enc)

        dec_out = self.conv_trans(dec_out)

        dec_out = self.projector(dec_out.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :N] # filter the covariates

        # 2024.4.1新增 begin

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # 2024.4.1新增 end
        dec_out = dec_out + trend
        return dec_out






