import torch.nn as nn
import torch

from layers.StandardNorm import Normalize



DECOMP_KERNELSIZE = {
    'ECL':[13 ,17,21],
    'WTH':[13 ,17,21],
    'Exchange':[13,17],
    'Traffic':[13,17,21,25],
    'ETTm2':[13,17],
    'ETTm1':[13,17],
    'ETTh2':[13,17],
    'ETTh1':[13,17],
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
        merge_list = torch.tensor([],device=self.device)
        for branch_item in branch_list:
            merge_list = torch.cat((merge_list, branch_item.permute(0, 2, 1).unsqueeze(1)), dim=1)

        merge_list = merge_list.permute(0, 3, 1, 2)
        merge_list_result = self.merge(merge_list).squeeze(-2)
        return merge_list_result



class Seasonal_Prediction(nn.Module):
    def __init__(self, configs,embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(Seasonal_Prediction, self).__init__()

        self.configs = configs
        self.pred_len = configs.pred_len
        self.channel_independence = configs.channel_independence
        self.e_layers = configs.e_layers
        # 消融 fourier
        if self.configs.use_x_mark_enc == 1:
            self.x_mark_len = self.configs.x_mark_len
        else:
            self.x_mark_len = 0

        self.use_fourier = configs.use_fourier
        self.use_space_merge = configs.use_space_merge
        self.merge_inner = Conv2dMergeBlock(in_channels=configs.d_model,out_channels=configs.d_model,num_branch=self.configs.down_sampling_layers + 1)
        self.up_sampling = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.d_model // (configs.down_sampling_window ** i),
                    configs.d_model,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in + self.x_mark_len, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

    def fourier_downsampling(self,signal, downsample_factor):
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
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            if self.use_fourier == 1:
                x_enc_sampling = self.fourier_downsampling(x_enc_ori, 2 ** (i+1))
            else:
                x_enc_sampling = down_pool(x_enc_ori) # 32*12*128  --- 32*12 * 64
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def up_sampling_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        x_list = x_list[0]
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            # dec_out = F.interpolate(enc_out.permute(0, 2, 1), size=self.configs.d_model, mode='linear', align_corners=False).permute(
            #     0, 2, 1)  # align temporal dimension
            dec_out = self.up_sampling[i](enc_out.permute(0, 2, 1)).permute(
                0, 2, 1)  # align temporal dimension
            dec_out = dec_out.reshape(B, self.x_mark_len + self.configs.enc_in, self.configs.d_model).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)
        return dec_out_list

    def forward(self, dec):
        dec  = dec.permute(0,2,1)  # 32* 128 * 12（7+5）  5.4改 32 96 128

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(dec,None)
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
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T,1)
                x_list.append(x)
        enc_out_list = x_list  # 384*128*1   384*64*1   384*32*1   384*16*1

        dec_out_list = self.up_sampling_mixing(B, enc_out_list, x_list)

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

