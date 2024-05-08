import torch.nn as nn
import torch

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


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(

                    torch.nn.Conv1d(
                        configs.d_model,
                        configs.d_model,
                        1
                    ),
                    nn.ReLU(),
                    torch.nn.Conv1d(
                        configs.d_model,
                        configs.d_model,
                        1,
                        stride=2
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )



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
            # out_low_res = self.down_sampling_layers[i](out_high)
            # out_low_res = self.season_fourier_downsampling(out_high, 2)
            out_low_res = self.down_sampling_layers[i](out_high)
            # out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res  # 能不能用注意力

            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.ConvTranspose1d(
                        configs.d_model,
                        configs.d_model,
                        1

                    ),
                    nn.ReLU(),
                    torch.nn.ConvTranspose1d(
                        configs.d_model,
                        configs.d_model,
                        kernel_size=2,
                        stride=2
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):

            out_high_res = self.up_sampling_layers[i](out_low)
            # out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        channel_independence = 1
        down_sampling_window = 2

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.down_sampling_window = configs.down_sampling_window

        self.decompsition = series_decomp(25)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer_test = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Conv1d(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.d_ff,
                        kernel_size=1
                    ),
                    nn.GELU(),
                    torch.nn.Conv1d(
                        configs.d_ff,
                        configs.seq_len // (configs.down_sampling_window ** i),
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
        season_list_copy = []
        trend_list_copy = []
        for x in x_list:
            # season, trend = x,x
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
            season_list_copy.append(season)
            trend_list_copy.append(trend)

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for i, ori, out_season, out_trend, length,sl,tl in zip(range(len(x_list)), x_list, out_season_list, out_trend_list,
                                                         length_list,season_list_copy,trend_list_copy):
            # out = out_season
            out = out_season + out_trend

            if self.channel_independence:
                # out = ori + self.out_cross_layer(out) # 使用卷积
                # out = ori + self.out_cross_layer(out)
                out = ori + self.out_cross_layer_test[i](out)
            out_list.append(out[:, :length, :])
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
        self.decomp_layer = [series_decomp(k) for k in self.decomp_kernel]
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.dataEmbedding_wo_pos = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                         configs.dropout)
        # 消融 fourier
        self.use_fourier = configs.use_fourier
        self.use_space_merge = configs.use_space_merge

        self.merge_inner = Conv2dMergeBlock(in_channels=configs.pred_len, out_channels=configs.pred_len,
                                            num_branch=self.configs.down_sampling_layers + 1)
        # self.merge_outer = Conv2dMergeBlock(in_channels=configs.d_model, out_channels=configs.d_model,
        #                                     num_branch=len(self.decomp_kernel))

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                    kernel_size=1
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.channel_independence == 1:

            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
    # 傅里叶下采样

    def fourier_downsampling(self, signal, downsample_factor):
        """
        使用傅里叶变换进行下采样
        Args:
        - signal: 输入的时域信号，形状为 [batch_size, seq_len, channels]
        - downsample_factor: 下采样因子，用于指定下采样的倍率
        Returns:
        - downsampled_signal: 下采样后的时域信号，形状为 [batch_size, downsampled_seq_len, channels]
        """
        batch_size,  channels,seq_len = signal.shape

        # 对信号进行傅里叶变换
        freq_signal = torch.fft.fft(signal,dim=1, norm='ortho')

        # 计算下采样后的频域信号长度
        downsampled_seq_len = seq_len // downsample_factor

        # 从高频到低频  2024.5.7
        # top_freq_indices = torch.topk(torch.abs(freq_signal), downsampled_seq_len, dim=0).indices
        # 仅保留频谱的前 downsampled_seq_len 个频率分量
        freq_signal_downsampled = freq_signal[:, :, :downsampled_seq_len]

        # freq_signal_downsampled = freq_signal[:,:,:downsampled_seq_len].real








        #
        # frequency_list = abs(freq_signal).mean(0).mean(-1)
        # frequency_list[0] = 0
        # _, top_list = torch.topk(abs(freq_signal), downsampled_seq_len)
        #
        # freq_signal_downsampled = freq_signal[:, :, top_list]

        # freq_signal_downsampled = torch.gather(freq_signal,dim=0,index=top_list)


        # _, top_list = torch.topk(abs(freq_signal).mean(0).mean(0), downsampled_seq_len)
        # #
        # freq_signal_downsampled = freq_signal[:, :, top_list] # 2024.5.7 17：26 这种取值会将每个batch的高值都定在同一个位置 是不是可以把8的维度mean  然后取【batch，seqlength】

        # print(freq_signal[0, 0, 49])
        # print(freq_signal[0, 0, 48])
        # print(freq_signal[0, 0, 50])
        # print(freq_signal[0, 0, 51])
        # print(freq_signal[0, 0, 47])
        # print(freq_signal[0, 0, 46])














        # _, top_list = torch.topk(abs(freq_signal).mean(1), downsampled_seq_len)
        # result_list = []
        # for i in range(batch_size):
        #     result_list.append(freq_signal[i, :, top_list[i]].unsqueeze(0))
        #
        # freq_signal_downsampled = torch.cat(result_list,dim=0)
        # freq_signal_downsampled = torch.gather(freq_signal, 0, top_list.unsqueeze(2)).squeeze(2)

        # 对下采样后的频域信号进行逆傅里叶变换
        downsampled_signal = torch.fft.ifft(freq_signal_downsampled, dim=1)

        # 取实部作为下采样后的时域信号
        downsampled_signal = downsampled_signal.real
        # result_list = []
        # for i in range(batch_size):
        #     result_list.append(signal[i, :, top_list[i]].unsqueeze(0))
        # #
        # freq_signal_downsampled = torch.cat(result_list, dim=0)
        return downsampled_signal

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
        x_mark_enc = x_mark_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            if self.use_fourier == 1:
                x_enc_sampling = self.fourier_downsampling(x_enc_ori, 2)
            else:
                x_enc_sampling = down_pool(x_enc_ori)  # 32*12*128  --- 32*12 * 64
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def projection_layer_fourier_downsampling(self, signal, downsample_factor):
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
        # downsampled_seq_len = channels // downsample_factor
        downsampled_seq_len = downsample_factor

        # 仅保留频谱的前 downsampled_seq_len 个频率分量
        freq_signal_downsampled = freq_signal[:, :, :downsampled_seq_len]

        # 对下采样后的频域信号进行逆傅里叶变换
        downsampled_signal = torch.fft.ifft(freq_signal_downsampled, dim=1)

        # 取实部作为下采样后的时域信号
        downsampled_signal = downsampled_signal.real

        return downsampled_signal

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                # dec_out = F.interpolate(enc_out.permute(0, 2, 1), size=self.configs.d_model, mode='linear', align_corners=False).permute(
                #     0, 2, 1)  # align temporal dimension
                # dec_out = self.predict_layers[i](enc_out) # align temporal dimension
                # dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
                dec_out = self.predict_layers[i](enc_out)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                # dec_out = self.projection_layer(dec_out)
                # if self.channel_independence == 1:
                #     dec_out = self.projection_layer_fourier_downsampling(dec_out,1)

                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def pre_enc(self, x_list):
        # if self.channel_independence == 1:
        return (x_list, None)
        # else:
        #     out1_list = []
        #     out2_list = []
        #     for x in x_list:
        #         x_1, x_2 = self.preprocess(x)
        #         out1_list.append(x_1)
        #         out2_list.append(x_2)
        #     return (out1_list, out2_list)

    def forward(self, dec, x_mark):
        dec = dec.permute(0, 2, 1)  # 32* 128 * 12（7+5）  5.4改 32 96 128
        x_mark = x_mark.permute(0, 2, 1)  # 32* 128 * 12（7+5）  5.4改 32 96 128

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
                x_list.append(x)

        # embedding  2024.5.4 把embedding放到list里面
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

        # enc_out_list = x_list  # 384*128*1   384*64*1   384*32*1   384*16*1

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

        return result




class TrendMultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(TrendMultiScaleSeasonMixing, self).__init__()
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

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

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
            # out_low_res = self.down_sampling_layers[i](out_high)
            # out_low_res = self.season_fourier_downsampling(out_high, 2)
            out_low_res = self.down_sampling_layers[i](out_high)
            # out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res  # 能不能用注意力

            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class TrendMultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(TrendMultiScaleTrendMixing, self).__init__()

        # self.up_sampling_layers = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             torch.nn.ConvTranspose1d(
        #                 configs.d_model,
        #                 configs.d_model,
        #                 1
        #
        #             ),
        #             nn.ReLU(),
        #             torch.nn.ConvTranspose1d(
        #                 configs.d_model,
        #                 configs.d_model,
        #                 kernel_size=2,
        #                 stride=2
        #             ),
        #         )
        #         for i in reversed(range(configs.down_sampling_layers))
        #     ])



        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])
    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)] # 保留最原始的low  维度96

        for i in range(len(trend_list_reverse) - 1):

            out_high_res = self.up_sampling_layers[i](out_low)
            # out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class TrendPastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(TrendPastDecomposableMixing, self).__init__()
        channel_independence = 1
        down_sampling_window = 2

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.down_sampling_window = configs.down_sampling_window

        self.decompsition = series_decomp(25)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = TrendMultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = TrendMultiScaleTrendMixing(configs)

        # 第二步  2.
        # self.out_cross_layer_test = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             torch.nn.Conv1d(
        #                 configs.seq_len * (configs.down_sampling_window ** i),
        #                 configs.d_ff,
        #                 kernel_size=1
        #             ),
        #             nn.GELU(),
        #             torch.nn.Conv1d(
        #                 configs.d_ff,
        #                 configs.seq_len * (configs.down_sampling_window ** i),
        #                 kernel_size=1
        #
        #             ),
        #
        #         )
        #         for i in reversed(range(configs.down_sampling_layers + 1))
        #     ]
        # )

        self.out_cross_layer_test = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

        # self.out_cross_layer_test = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             torch.nn.Conv1d(
        #                 configs.seq_len // (configs.down_sampling_window ** i),
        #                 configs.d_ff,
        #                 kernel_size=1
        #             ),
        #             nn.GELU(),
        #             torch.nn.Conv1d(
        #                 configs.d_ff,
        #                 configs.seq_len // (configs.down_sampling_window ** i),
        #                 kernel_size=1
        #
        #             ),
        #
        #         )
        #         for i in range(configs.down_sampling_layers + 1)
        #     ]
        # )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        season_list_copy = []
        trend_list = []
        trend_list_copy = []
        for x in x_list:
            # season, trend = x,x
            season, trend = self.decompsition(x)
            # season, trend = x,x
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
            season_list_copy.append(season)
            trend_list_copy.append(trend)

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for i, ori, out_season, out_trend, length,sl,tl in zip(range(len(x_list)), x_list, out_season_list, out_trend_list,
                                                         length_list,season_list_copy,trend_list_copy):
            # out = out_season
            # out = out_season + tl
            # out = out_season + out_trend
            out = out_trend + sl

            if self.channel_independence:
                # out = ori + self.out_cross_layer(out) # 使用卷积
                out = ori + self.out_cross_layer_test(out)
                # out = ori + self.out_cross_layer_test[i](out)
            out_list.append(out[:, :length, :])
        return out_list




class Trend_Prediction(nn.Module):
    def __init__(self,configs,embedding_size,n_heads,dropout,d_layers,decomp_kernel,c_out,conv_kernel,isometric_kernel,device):
        super(Trend_Prediction, self).__init__()
        self.configs = configs
        self.decompsition = series_decomp(25)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        # 消融 fourier
        self.use_fourier = configs.use_fourier
        self.use_space_merge = configs.use_space_merge
        self.e_layers = configs.e_layers
        self.channel_independence = configs.channel_independence
        self.pred_len = configs.pred_len

        self.merge_inner = Conv2dMergeBlock(in_channels=configs.pred_len, out_channels=configs.pred_len,
                                            num_branch=self.configs.down_sampling_layers + 1)

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        self.dataEmbedding_wo_pos = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                         configs.dropout)
        if self.channel_independence == 1:
            # self.projection_layer = nn.Conv1d(in_channels= configs.pred_len,out_channels= configs.pred_len,kernel_size=128,stride=1,padding=0)
            # self.projection_layer = nn.Conv1d(in_channels= configs.pred_len,out_channels= configs.pred_len,kernel_size=128,stride=1,padding=0)
            # self.projection_layer = nn.MaxPool1d(kernel_size=configs.d_model, return_indices=False)
            # self.projection_layer = nn.AdaptiveAvgPool1d(output_size=1)
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)

        self.trend_pdm_blocks = nn.ModuleList([TrendPastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        # 第一步  1.
        # self.predict_layers = torch.nn.ModuleList(
        #     [
        #         torch.nn.Conv1d(
        #             configs.seq_len * (configs.down_sampling_window ** i),
        #             configs.pred_len,
        #             kernel_size=1
        #         )
        #         for i in reversed(range(configs.down_sampling_layers + 1))
        #     ]
        # )

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                    kernel_size=1
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        # self.trend_up_sampling = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             torch.nn.ConvTranspose1d(
        #                 configs.enc_in,
        #                 configs.enc_in,
        #                 1
        #
        #             ),
        #             nn.ReLU(),
        #             torch.nn.ConvTranspose1d(
        #                 configs.enc_in,
        #                 configs.enc_in,
        #                 kernel_size=2 ,
        #                 stride=2
        #             ),
        #         )
        #         for i in range(configs.down_sampling_layers)
        #     ])


    def fourier_downsampling(self, signal, downsample_factor):
        """
        使用傅里叶变换进行下采样
        Args:
        - signal: 输入的时域信号，形状为 [batch_size, seq_len, channels]
        - downsample_factor: 下采样因子，用于指定下采样的倍率
        Returns:
        - downsampled_signal: 下采样后的时域信号，形状为 [batch_size, downsampled_seq_len, channels]
        """
        batch_size,  channels,seq_len = signal.shape

        # 对信号进行傅里叶变换
        freq_signal = torch.fft.fft(signal,dim=1, norm='ortho')

        # 计算下采样后的频域信号长度
        downsampled_seq_len = seq_len // downsample_factor

        # 从低频到高频  2024.5.7
        # top_freq_indices = torch.topk(torch.abs(freq_signal), downsampled_seq_len, dim=1,largest=False).indices

        # 仅保留频谱的前 top_freq_indices 个频率分量
        # freq_signal_downsampled = freq_signal[:, :, :top_freq_indices]
        # freq_signal_downsampled = freq_signal[:, :, :downsampled_seq_len]









        # frequency_list = abs(freq_signal).mean(0).mean(-1)
        # freq1 =abs(freq_signal).mean(0)
        # freq2 = abs(freq_signal).mean(1)
        # freq3 = abs(freq_signal).mean(2)
        # frequency_list[0] = 0
        # _, top_list = torch.topk(abs(freq_signal).mean(0).mean(0), downsampled_seq_len,largest=False)
        # #
        # freq_signal_downsampled = freq_signal[:, :, top_list] # 2024.5.7 17：26 这种取值会将每个batch的高值都定在同一个位置

        # print(freq_signal[0, 0, 49])
        # print(freq_signal[0, 0, 48])
        # print(freq_signal[0, 0, 50])
        # print(freq_signal[0, 0, 51])
        # print(freq_signal[0, 0, 47])
        # print(freq_signal[0, 0, 46])




        # freq_signal_downsampled = torch.gather(freq_signal,dim=2,index=top_list)

        freq_signal_downsampled = freq_signal[:,:,downsampled_seq_len:].real


        # _, top_list = torch.topk(abs(freq_signal).mean(1), downsampled_seq_len,largest=True )

        # result_list = []
        # for i in range(batch_size):
        #     result_list.append(freq_signal[i, :, top_list[i]].unsqueeze(0))
        #
        # freq_signal_downsampled = torch.cat(result_list,dim=0)

        # freq_signal_downsampled = torch.gather(freq_signal, 0, top_list.unsqueeze(1)).squeeze(1)

        # 对下采样后的频域信号进行逆傅里叶变换
        # downsampled_signal = torch.fft.ifft(freq_signal, dim=1)

        # 取实部作为下采样后的时域信号
        # downsampled_signal = downsampled_signal.real

        # result_list = []
        # for i in range(batch_size):
        #     result_list.append(downsampled_signal[i, :, top_list[i]].unsqueeze(0))

        # freq_signal_downsampled = torch.cat(result_list, dim=0)
        # freq_signal_downsampled = torch.gather(downsampled_signal, 0, top_list.unsqueeze(1)).squeeze(1)
        # freq_signal_downsampled = torch.gather(signal, 0, top_list.unsqueeze(1)).squeeze(1)  # 或者直接取siginal


        return freq_signal_downsampled


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
        if x_mark_enc is not None:
            x_mark_enc = x_mark_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            if self.use_fourier == 1:
                x_enc_sampling = self.fourier_downsampling(x_enc_ori, 2)
                # x_enc_sampling = self.trend_up_sampling[i](x_enc_ori)



            else:
                x_enc_sampling = down_pool(x_enc_ori)  # 32*12*128  --- 32*12 * 64
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc
    def pre_enc(self, x_list):
        # if self.channel_independence == 1:
        return (x_list, None)
        # else:
        #     out1_list = []
        #     out2_list = []
        #     for x in x_list:
        #         x_1, x_2 = self.preprocess(x)
        #         out1_list.append(x_1)
        #         out2_list.append(x_2)
        #     return (out1_list, out2_list)

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, dec, x_mark):

        dec = dec.permute(0, 2, 1)  # 32* 128 * 12（7+5）  5.4改 32 96 128
        if x_mark is not None:
            x_mark = x_mark.permute(0, 2, 1)  # 32* 128 * 12（7+5）  5.4改 32 96 128

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
                x_list.append(x)

        # embedding  2024.5.4 把embedding放到list里面
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
        enc_out_list_reverse = enc_out_list.copy()
        # 第三步  3.

        # enc_out_list_reverse.reverse()
        for j in range(self.e_layers):
            enc_out_list_reverse = self.trend_pdm_blocks[j](enc_out_list_reverse)

        x_list_0_reverse = x_list[0].copy()
        # 第四步  4.

        # x_list_0_reverse.reverse()
        x_list_reverse = (x_list_0_reverse,None)
        dec_out_list = self.future_multi_mixing(B, enc_out_list_reverse, x_list_reverse)

        if self.use_space_merge == 1:
            dec_out = self.merge_inner(dec_out_list)
        else:
            dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)

        return dec_out




