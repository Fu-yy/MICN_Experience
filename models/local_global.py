import torch.nn as nn
import torch
import torch.nn.functional as F

from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

DECOMP_KERNELSIZE = {
    'ECL': [13, 17, 21],
    'WTH': [13, 17, 21],
    'Exchange': [13, 17],
    'Traffic': [13, 17, 21, 25],
    'ETTm2': [13, 17],
    'ETTm1': [13, 17],
    'ETTh2': [13, 17],
    'ETTh1': [13, 17],
}
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# 2024.4.2 用autoformer的老multi替换MICN的multi
class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)

            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class Conv2dMergeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_branch):
        super(Conv2dMergeBlock, self).__init__()
        self.merge = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(num_branch, 1))
        self.device = 'cuda:0'

    def forward(self, branch_list):
        merge_list = torch.tensor([], device=self.device)
        for branch_item in branch_list:
            merge_list = torch.cat((merge_list, branch_item.permute(0, 2, 1).unsqueeze(1)), dim=1)

        merge_list = merge_list.permute(0, 3, 1, 2)
        merge_list_result = self.merge(merge_list).squeeze(-2)
        return merge_list_result


# class FourierLow(nn.Module):
#     def __init__(self, seq_len ,individual,enc_in,cut_freq):
#         super(FourierLow, self).__init__()
#         self.seq_len = seq_len
#         self.channels = enc_in
#         self.dominance_freq = cut_freq
#         self.n_fft = self.seq_len // 2 + 1  # FFT输出的大小
#         self.individual = individual
#
#         # 注意：为了处理复数数据，我们的频率上采样层的输入和输出尺寸都翻倍
#         if self.individual:
#             self.freq_upsampler = nn.ModuleList([nn.Linear(self.n_fft * 2, self.n_fft * 2, bias=False) for _ in range(self.channels)])
#         else:
#             self.freq_upsampler = nn.Linear(self.n_fft * 2 * self.channels, self.n_fft * 2 * self.channels, bias=False)
#
#     def forward(self, x):
#         x_mean = torch.mean(x, dim=1, keepdim=True)
#         x_normalized = (x - x_mean) / torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
#
#         # 执行FFT变换
#         low_specx = torch.fft.rfft(x_normalized, dim=1)
#         low_specx[:, self.dominance_freq:, :] = 0  # 应用LPF
#
#         # 拆分实部和虚部
#         real_part = low_specx.real
#         imag_part = low_specx.imag
#
#         # 将实部和虚部拼接在一起形成实数张量
#         low_specx_combined = torch.cat([real_part, imag_part], dim=-1)
#
#         # 应用全连接层之后，假设low_specxy_combined已经是正确的形状，其中包含了合并的实部和虚部
#         # low_specxy_combined的形状应该是 (batch_size, self.seq_len // 2 + 1, 2 * self.channels)
#
#         if isinstance(self.freq_upsampler, nn.ModuleList):
#             low_specxy_combined = torch.stack([
#                 self.freq_upsampler[i](low_specx_combined[:, :, i].view(-1, 2 * self.n_fft))
#                 for i in range(self.channels)
#             ], dim=-1).view(-1, self.n_fft, 2)
#         else:
#             low_specxy_combined = self.freq_upsampler(low_specx_combined.view(-1, self.n_fft * 2 * self.channels))
#             # 确保low_specxy_combined回到期望的形状
#             low_specxy_combined = low_specxy_combined.view(-1, self.n_fft, 2 * self.channels)
#
#         # 分割实部和虚部，需要考虑channels维度
#         real_part, imag_part = torch.split(low_specxy_combined, self.channels, dim=-1)
#
#         # 将real_part和imag_part的形状调整为复数操作所需的形状
#         real_part = real_part.view(-1, self.seq_len // 2 + 1, self.channels)
#         imag_part = imag_part.view(-1, self.seq_len // 2 + 1, self.channels)
#
#         # 重新组合为复数张量
#         low_specxy_ = torch.complex(real_part, imag_part)
#
#         # 接下来的代码保持不变
#         low_xy = torch.fft.irfft(low_specxy_, n=self.seq_len, dim=1)
#         xy = (low_xy * torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)) + x_mean
#
#         return xy


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.configs = configs
        if configs.season_use_fourier == 0:
            self.down_sampling_layers = nn.Sequential(
                        torch.nn.Conv1d(
                            configs.seq_len,
                            configs.seq_len,
                            kernel_size=1
                        ),
                        nn.ReLU(),
                        torch.nn.Conv1d(
                            configs.seq_len,
                            configs.seq_len,
                            kernel_size=1

                        ),

                    )


            # self.down_sampling_layers = torch.nn.ModuleList(
            #     [
            #         nn.Sequential(
            #             torch.nn.Linear(
            #                 configs.seq_len // (configs.down_sampling_window ** i),
            #                 configs.seq_len // (configs.down_sampling_window ** (i + 1)),
            #             ),
            #             nn.GELU(),
            #             torch.nn.Linear(
            #                 configs.seq_len // (configs.down_sampling_window ** (i + 1)),
            #                 configs.seq_len // (configs.down_sampling_window ** (i + 1)),
            #             ),
            #
            #         )
            #         for i in range(configs.down_sampling_layers)
            #     ]
            # )
        # self.down_sampling_layers = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #
        #             torch.nn.Conv1d(
        #                 configs.d_model,
        #                 configs.d_model,
        #                 1
        #             ),
        #             nn.ReLU(),
        #             torch.nn.Conv1d(
        #                 configs.d_model,
        #                 configs.d_model,
        #                 1,
        #                 stride=2
        #             ),
        #
        #         )
        #         for i in range(configs.down_sampling_layers)
        #     ]
        # )
        # self.fourier_layers = torch.nn.ModuleList(
        #     [
        #         FourierLow(seq_len=configs.seq_len // (configs.down_sampling_window ** i),individual=False,enc_in=configs.d_model,cut_freq=50)
        #         for i in range(configs.down_sampling_layers +1)
        #     ]
        # )

    def season_fourier_downsampling(self, signal, downsample_factor):
        """
        使用傅里叶变换进行下采样
        Args:
        - signal: 输入的时域信号，形状为 [batch_size, seq_len, channels]
        - downsample_factor: 下采样因子，用于指定下采样的倍率
        Returns:
        - downsampled_signal: 下采样后的时域信号，形状为 [batch_size, downsampled_seq_len, channels]
        """
        batch_size, seq_len, channels = signal.shape

        # 对信号进行傅里叶变换
        freq_signal = torch.fft.fft(signal, dim=1, norm='ortho')

        # 计算下采样后的频域信号长度
        downsampled_seq_len = channels // downsample_factor

        # 仅保留频谱的前 downsampled_seq_len 个频率分量
        freq_signal_downsampled = freq_signal[:, :, :downsampled_seq_len]

        # 对下采样后的频域信号进行逆傅里叶变换
        downsampled_signal = torch.fft.ifft(freq_signal_downsampled, dim=1)

        # 取实部作为下采样后的时域信号
        downsampled_signal = downsampled_signal.real

        return downsampled_signal

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            if self.configs.season_use_fourier == 0:
                out_low_res = self.down_sampling_layers(out_high.permute(0,2,1)).permute(0,2,1)
            else:
                out_low_res = self.season_fourier_downsampling(out_high, 2)

            # out_low_res = self.down_sampling_layers[i](out_high.permute(0,2,1)).permute(0,2,1)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list

    # def forward(self,season_list):
    #     result_list = []
    #     for i,season_item in zip(range(len(season_list)),season_list):
    #         # season_item_f = self.fourier_layers[i](season_item.permute(0, 2, 1))
    #         result_list.append(season_item.permute(0, 2, 1))
    #     return result_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        self.configs = configs
        self.up_sampling_layers =nn.Sequential(
                        torch.nn.Conv1d(
                            configs.d_model,
                            configs.d_model,
                            kernel_size=1
                        ),
                        nn.ReLU(),
                        torch.nn.Conv1d(
                            configs.d_model,
                            configs.d_model,
                            kernel_size=1

                        ),

                    )



        # if configs.trend_use_conv == 1:
        #     self.up_sampling_layers = torch.nn.ModuleList(
        #         [
        #             nn.Sequential(
        #                 torch.nn.ConvTranspose1d(
        #                     configs.d_model,
        #                     configs.d_model,
        #                     1
        #
        #                 ),
        #                 nn.ReLU(),
        #                 torch.nn.ConvTranspose1d(
        #                     configs.d_model,
        #                     configs.d_model,
        #                     kernel_size=2,
        #                     stride=2
        #                 ),
        #             )
        #             for i in reversed(range(configs.down_sampling_layers))
        #         ])
        # else:
        #     self.up_sampling_layers = torch.nn.ModuleList(
        #         [
        #             nn.Sequential(
        #                 torch.nn.Linear(
        #                     configs.seq_len // (configs.down_sampling_window ** (i + 1)),
        #                     configs.seq_len // (configs.down_sampling_window ** i),
        #                 ),
        #                 nn.GELU(),
        #                 torch.nn.Linear(
        #                     configs.seq_len // (configs.down_sampling_window ** i),
        #                     configs.seq_len // (configs.down_sampling_window ** i),
        #                 ),
        #             )
        #             for i in reversed(range(configs.down_sampling_layers))
        #         ])

    # def forward(self, trend_list):
    #
    #     # mixing low->high
    #     trend_list_reverse = trend_list.copy()
    #     trend_list_reverse.reverse()
    #     out_low = trend_list_reverse[0]
    #     out_high = trend_list_reverse[1]
    #     out_trend_list = [out_low.permute(0, 2, 1)]
    #
    #     for i in range(len(trend_list_reverse) - 1):
    #         if self.configs.trend_use_conv == 1:
    #             out_high_res = self.up_sampling_layers[i](out_low.permute(0, 2, 1)).permute(0, 2, 1)
    #             # out_high_res = self.up_sampling_layers[i](out_low)
    #
    #         else:
    #             # out_high_res = self.up_sampling_layers[i](out_low)
    #             out_high_res = self.up_sampling_layers(out_low)
    #
    #
    #         out_high = out_high + out_high_res
    #         out_low = out_high
    #         if i + 2 <= len(trend_list_reverse) - 1:
    #             out_high = trend_list_reverse[i + 2]
    #         out_trend_list.append(out_low.permute(0, 2, 1))
    #
    #
    #
    #
    #
    #     out_trend_list.reverse()
    #     return out_trend_list
    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            if self.configs.trend_use_conv == 1:
                out_high_res = self.up_sampling_layers[i](out_low.permute(0, 2, 1)).permute(0, 2, 1)
                # out_high_res = self.up_sampling_layers[i](out_low)

            else:
                # out_high_res = self.up_sampling_layers[i](out_low)
                out_high_res = self.up_sampling_layers(out_low)

            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list
    # def forward(self, season_list):
    #     result_list = []
    #     for i in season_list:
    #         result_list.append(i.permute(0, 2, 1))
    #     return result_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.down_sampling_window = configs.down_sampling_window

        self.decompsition = series_decomp(49)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence


        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer_test = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Conv1d(
                        configs.d_model,
                        configs.d_ff,
                        kernel_size=1
                    ),
                    nn.ReLU(),
                    torch.nn.Conv1d(
                        configs.d_ff,
                        configs.d_model,
                        kernel_size=1

                    ),

                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            # season, trend = self.decompsition(x)
            # _, trend = self.decompsition(x)

            season,trend = x,x
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)

            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        # 新增将1变成d_model

        # bottom-up season mixin
        out_season_list =season_list
        # out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for i, ori, out_season, out_trend, length in zip(range(len(x_list)), x_list, out_season_list, out_trend_list,
                                                         length_list):
            # out = out_trend
            # out = out_season + out_trend
            # if self.channel_independence:
            out = out_trend + self.out_cross_layer_test[i](out_trend.permute(0, 2, 1)).permute(0, 2, 1)
            # out = ori + self.out_cross_layer_test[i](out)
            # out = ori + self.out_cross_layer(out)
            # out_list.append(out[:, :length, :])
            out_list.append(out)
        return out_list


class Seasonal_Prediction(nn.Module):
    def __init__(self, configs, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                 conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(Seasonal_Prediction, self).__init__()
        self.decomp_kernel = DECOMP_KERNELSIZE.get(configs.data)

        print("实际的DECOMP:")
        print(self.decomp_kernel)
        self.configs = configs
        self.pred_len = configs.pred_len
        self.channel_independence = configs.channel_independence
        self.e_layers = configs.e_layers
        # self.decomp_layer = [series_decomp(k) for k in self.decomp_kernel]
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])
        self.dataEmbedding_wo_pos = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                         configs.dropout)

        self.fourier_layer = FourierLow(seq_len=configs.seq_len, individual=False, enc_in=configs.enc_in,
                                        cut_freq=configs.cut_freq)

        # 消融 fourier
        self.use_fourier = configs.use_fourier
        self.use_space_merge = configs.use_space_merge

        self.merge_inner = Conv2dMergeBlock(in_channels=configs.pred_len, out_channels=configs.pred_len,
                                            num_branch=self.configs.down_sampling_layers + 1)

        if configs.pred_use_conv == 1:
            self.predict_layers = torch.nn.ModuleList(
                [

                    torch.nn.ConvTranspose1d(
                        configs.d_model,
                        configs.d_model,
                        kernel_size=2 ** i,
                        stride=2 ** i
                    )

                    for i in range(configs.down_sampling_layers + 1)
                ])
        else:
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len ,
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)


        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True,
                          non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )


    # 傅里叶下采样

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc

        # B,T,C -> B,C,T
        period_list, period_weight_list = FFT_for_Period(x_enc,self.configs.down_sampling_layers)
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)


        for i,period,period_weight in zip(range(self.configs.down_sampling_layers),period_list,period_weight_list):
            if self.use_fourier == 1:
                # x_enc_sampling = self.fourier_downsampling(x_enc_ori,2)


                # x_enc_sampling = self.fourier_downsampling(x_enc_ori, 2 ** (i + 1))
                x_enc_sampling = self.fourier_layer(x_enc_ori,period)

                # period_weight = F.softmax(period_weight)
                # x_enc_sampling = period_weight * x_enc_sampling

            else:
                x_enc_sampling = down_pool(x_enc_ori)  # 32*12*128  --- 32*12 * 64
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            # x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori)
                x_mark_enc_mark_ori = x_mark_enc_mark_ori

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                # dec_out = F.interpolate(enc_out.permute(0, 2, 1), size=self.configs.d_model, mode='linear', align_corners=False).permute(
                #     0, 2, 1)  # align temporal dimension
                # dec_out = self.predict_layers[i](enc_out)  # align temporal dimension
                dec_out = self.projection_layer(enc_out)
                dec_out = self.predict_layers[i](dec_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension


                # dec_out = self.predict_layers[i](enc_out)# align temporal dimension

                dec_out = dec_out.reshape(B, self.configs.c_out, self.configs.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, dec, x_mark):
        # dec = dec.permute(0, 2, 1)  # 32* 128 * 12（7+5）  5.4改 32 96 128

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(dec, x_mark)
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

                x_list.append(self.up_to_d_model(x))

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.dataEmbedding_wo_pos(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.dataEmbedding_wo_pos(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        for j in range(self.e_layers):
            enc_out_list = self.pdm_blocks[j](enc_out_list)
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        if self.use_space_merge == 1:
            dec_out = self.merge_inner(dec_out_list)
        else:
            dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)

        # t = multi
        # merge
        # result = torch.stack(multi, dim=-1).sum(-1)
        # result = self.merge_outer(multi)
        result = self.normalize_layers[0](dec_out, 'denorm')
        # result = self.fourier_layer(result)

        return result



class FourierLow(nn.Module):
    def __init__(self, seq_len, individual, enc_in, cut_freq):
        super(FourierLow, self).__init__()
        self.seq_len = seq_len
        self.channels = enc_in
        self.dominance_freq = cut_freq
        self.n_fft = self.seq_len // 2 + 1  # FFT输出的大小
        self.individual = individual

        # 注意：为了处理复数数据，我们的频率上采样层的输入和输出尺寸都翻倍

        # self.freq_upsampler = nn.Conv1d(self.n_fft , self.n_fft ,kernel_size=1)
        #
        # self.freq_upsampler = FourierLowEinsum(self.channels)
        self.freq_upsampler = nn.Linear(self.channels * 2, self.channels * 2, bias=False)

        self.layer_norm = nn.LayerNorm(self.seq_len)
        self.activate = nn.GELU()

    def forward(self, x,dominance_freq):
        x = x.permute(0,2,1)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_normalized = (x - x_mean) / torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        # x_normalized = x

        # 执行FFT变换
        low_specx = torch.fft.rfft(x_normalized, dim=1)
        low_specx[:, dominance_freq:, :] = 0  # 应用LPF

        # 拆分实部和虚部
        real_part = low_specx.real
        imag_part = low_specx.imag

        # 将实部和虚部拼接在一起形成实数张量
        low_specx_combined = torch.cat([real_part, imag_part], dim=-1)

        # 应用全连接层之后，假设low_specxy_combined已经是正确的形状，其中包含了合并的实部和虚部
        # low_specxy_combined的形状应该是 (batch_size, self.seq_len // 2 + 1, 2 * self.channels)

        # low_specx_combined = low_specx_combined.view(-1, self.n_fft * 2 * self.channels)
        low_specxy_combined = self.freq_upsampler(low_specx_combined)


        # 确保low_specxy_combined回到期望的形状
        # low_specxy_combined = low_specxy_combined.view(-1, self.n_fft, 2 * self.channels)

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
        xy = xy.permute(0,2,1)
        # return self.layer_norm(self.activate(xy))
        return xy
