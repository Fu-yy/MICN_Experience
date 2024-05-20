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
        print("MICN中的DECOMP:")
        print(configs.decomp_kernel)
        # embedding

        if self.configs.use_invertembed == 1:
            self.inverted_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            # self.inverted_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
            #                                           configs.dropout)
        else:
            self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        self.conv_trans = Seasonal_Prediction(configs=configs,embedding_size=d_model, n_heads=n_heads, dropout=dropout,
                                     d_layers=d_layers, decomp_kernel=self.decomp_kernel, c_out=c_out, conv_kernel=conv_kernel,
                                     isometric_kernel=isometric_kernel, device=device)

        self.regression = nn.Linear(seq_len, out_len)
        self.regression.weight = nn.Parameter((1/out_len) * torch.ones([out_len, seq_len]), requires_grad=True)


        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # 2024.4.1新增 begin

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        # x_enc = self.fourier_layer(x_enc)
        _, _, N = x_enc.shape # B L N

        # 2024.4.1新增 end

        # trend-cyclical prediction block: regre or mean
        if self.mode == 'regre':
            seasonal_init_enc, trend = self.decomp_multi(x_enc)
            # x_mark_enc_seasonal_init_enc, _ = self.decomp_multi(x_mark_enc)
            trend = self.regression(trend.permute(0,2,1)).permute(0, 2, 1)

        elif self.mode == 'mean':
            decomp_mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            # x_mark_enc_seasonal_init_enc, _ = self.decomp_multi(x_mark_enc)

            seasonal_init_enc, trend = self.decomp_multi(x_enc)
            trend = torch.cat([trend[:, -self.seq_len:, :], decomp_mean], dim=1)

        # embedding  2024.4.1 删除拼接
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        # seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1)

        seasonal_init_dec =seasonal_init_enc

        _,_, ss =seasonal_init_dec.shape
        _,_, xh =x_mark_enc.shape

        # if self.configs.use_invertembed == 1:
        #     # dec_out = self.inverted_embedding(seasonal_init_dec,x_mark_dec)
        #     dec_out = self.inverted_embedding(seasonal_init_dec,x_mark_enc)
        #     # dec_out = self.inverted_embedding(seasonal_init_dec,x_mark_enc)
        # else:
        #     dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)

        dec_out = self.conv_trans(seasonal_init_dec,x_mark_enc)
        # dec_out = self.conv_trans(dec_out)



        # dec_out = self.projector(dec_out.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :N] # filter the covariates

        # 2024.4.1新增 begin

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # 2024.4.1新增 end
        dec_out = dec_out + trend
        # dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        return dec_out



class FourierLow(nn.Module):
    def __init__(self, seq_len ,individual,enc_in,cut_freq):
        super(FourierLow, self).__init__()
        self.seq_len = seq_len
        self.channels = enc_in
        self.dominance_freq = cut_freq
        self.n_fft = self.seq_len // 2 + 1  # FFT输出的大小
        self.individual = individual

        # 注意：为了处理复数数据，我们的频率上采样层的输入和输出尺寸都翻倍
        if self.individual:
            self.freq_upsampler = nn.ModuleList([nn.Linear(self.n_fft * 2, self.n_fft * 2, bias=False) for _ in range(self.channels)])
        else:
            self.freq_upsampler = nn.Linear(self.n_fft * 2 * self.channels, self.n_fft * 2 * self.channels, bias=False)

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_normalized = (x - x_mean) / torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        # x_normalized = x


        # 执行FFT变换
        low_specx = torch.fft.rfft(x_normalized, dim=1)
        low_specx[:, :-self.dominance_freq, :] = 0  # 应用LPF

        # 拆分实部和虚部
        real_part = low_specx.real
        imag_part = low_specx.imag

        # 将实部和虚部拼接在一起形成实数张量
        low_specx_combined = torch.cat([real_part, imag_part], dim=-1)

        # 应用全连接层之后，假设low_specxy_combined已经是正确的形状，其中包含了合并的实部和虚部
        # low_specxy_combined的形状应该是 (batch_size, self.seq_len // 2 + 1, 2 * self.channels)

        if isinstance(self.freq_upsampler, nn.ModuleList):
            low_specxy_combined = torch.stack([
                self.freq_upsampler[i](low_specx_combined[:, :, i].view(-1, 2 * self.n_fft))
                for i in range(self.channels)
            ], dim=-1).view(-1, self.n_fft, 2)
        else:
            low_specxy_combined = self.freq_upsampler(low_specx_combined.view(-1, self.n_fft * 2 * self.channels))
            # 确保low_specxy_combined回到期望的形状
            low_specxy_combined = low_specxy_combined.view(-1, self.n_fft, 2 * self.channels)

        # 分割实部和虚部，需要考虑channels维度
        real_part, imag_part = torch.split(low_specxy_combined, self.channels, dim=-1)

        # 将real_part和imag_part的形状调整为复数操作所需的形状
        real_part = real_part.view(-1, self.seq_len // 2 + 1, self.channels)
        imag_part = imag_part.view(-1, self.seq_len // 2 + 1, self.channels)

        # 重新组合为复数张量
        low_specxy_ = torch.complex(real_part, imag_part)

        # 接下来的代码保持不变
        low_xy = torch.fft.irfft(low_specxy_, n=self.seq_len, dim=1)
        xy = (low_xy * torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)) + x_mean
        # xy = low_xy
        return xy



