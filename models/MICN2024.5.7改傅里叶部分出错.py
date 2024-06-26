import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_wo_pos
from models.embed import DataEmbedding, DataEmbedding_inverted
from models.local_global import Seasonal_Prediction, series_decomp_multi, Trend_Prediction


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
        print("MICN中的DECOMP:")
        print(configs.decomp_kernel)
        # embedding

        if self.configs.use_invertembed == 1:
            # self.inverted_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
            #                                         configs.dropout)
            self.inverted_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        self.conv_trans = Seasonal_Prediction(configs=configs,embedding_size=d_model, n_heads=n_heads, dropout=dropout,
                                     d_layers=d_layers, decomp_kernel=self.decomp_kernel, c_out=c_out, conv_kernel=conv_kernel,
                                     isometric_kernel=isometric_kernel, device=device)

        # self.trend_conv = Trend_Prediction(configs=configs, embedding_size=d_model, n_heads=n_heads, dropout=dropout,
        #                                       d_layers=d_layers, decomp_kernel=self.decomp_kernel, c_out=c_out,
        #                                       conv_kernel=conv_kernel,
        #                                       isometric_kernel=isometric_kernel, device=device)

        self.regression = nn.Linear(seq_len, self.pred_len)
        self.regression.weight = nn.Parameter((1/out_len) * torch.ones([out_len, seq_len]), requires_grad=True)

        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # 2024.4.1新增 begin

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N

        # 2024.4.1新增 end

        # trend-cyclical prediction block: regre or mean
        if self.mode == 'regre':
            seasonal_init_enc, trend = self.decomp_multi(x_enc)
            # x_mark_enc_seasonal_init_enc, _ = self.decomp_multi(x_mark_enc)
            # trend = self.regression(trend.permute(0,2,1)).permute(0, 2, 1)
        else:
            decomp_mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            # x_mark_enc_seasonal_init_enc, _ = self.decomp_multi(x_mark_enc)

            seasonal_init_enc, trend = self.decomp_multi(x_enc)
            trend = torch.cat([trend[:, -self.seq_len:, :], decomp_mean], dim=1)


        # 2024.5.5 对trend操作 begin：


        # trend_out = self.inverted_embedding(trend,x_mark_enc)

        # trend_result = self.trend_conv(trend,None)
        trend_result = trend

        # 2024.5.5 对trend操作 end：





        # embedding  2024.4.1 删除拼接
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        # seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1)

        seasonal_init_dec = seasonal_init_enc

        _,_, ss =seasonal_init_dec.shape
        _,_, xh =x_mark_enc.shape

        # if self.configs.use_invertembed == 1:
        #     # dec_out = self.inverted_embedding(seasonal_init_dec,x_mark_dec)
        #     dec_out = self.inverted_embedding(seasonal_init_dec,x_mark_enc)
        #     # dec_out = self.inverted_embedding(seasonal_init_dec,x_mark_enc)
        # else:
        #     dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)

        dec_out = self.conv_trans(seasonal_init_dec,x_mark_enc)




        # 2024.4.1新增 begin
        stdev_season,stdev_trend = stdev,stdev
        means_season, means_trend = means,means
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev_season[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means_season[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

            # trend_result = trend_result * (stdev_trend[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            # trend_result = trend_result + (means_trend[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))




        # 2024.4.1新增 end
        dec_out = dec_out + trend_result
        # dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        return dec_out






