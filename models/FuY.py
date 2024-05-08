import math
import os
import sys

# import einops
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
# from einops import rearrange

from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import ProbAttention, AttentionLayer, FlowAttention, FlashAttention, FullAttention

eni_path = os.path.dirname(__file__) + os.sep + "utils" + os.sep + "einops-0.7.0"
# print(eni_path)
sys.path.append(eni_path)
import einops
from einops import rearrange,repeat,reduce


class FuY(nn.Module):
    '''
    FuY
    '''

    def __init__(self, configs):
        super(FuY, self).__init__()
        self.configs = configs
        self.fgn = FGN(configs)

        # self.myInception = MyInceptionNet(configs)

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        fgn = self.fgn(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 32*96*7
        # myInception = self.myInception(x)
        return fgn


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)  # 4*13*32   24/2+1  时序降维
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]  # 返回周期及平均振幅  这几列top频率的平均值


class ConvolutionFilter(nn.Module):
    def __init__(self, kernel_size, d_model, stride, device='cuda'):
        super(ConvolutionFilter, self).__init__()
        self.kernel_size = kernel_size
        self.device = device
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2, bias=False, device=device)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 32 192 512

        x = self.conv(x)
        x = x.permute(0, 2, 1)  # 32 192 512

        return x


class ConvolutionFilter_512(nn.Module):
    def __init__(self, kernel_size, d_model, stride, device='cuda'):
        super(ConvolutionFilter_512, self).__init__()
        self.kernel_size = kernel_size
        self.device = device
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2, bias=False, device=device)

    #     self.act = nn.ReLU()
    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 32 192 512

        x = self.conv(x)
        x = x.permute(0, 2, 1)  # 32 192 512
        return x


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
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2,
                                    1)  # 32 6 512   # 32 6 7 # 取第二维度第一列 中间的维度是重复了6次 (self.kernel_size - 1) // 2  kernel_size = 13
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # 32 6 512  # 取第二维度最后一列  重复6次
        x = torch.cat([front, x, end], dim=1)  # 32 6+192+6  512
        x = self.avg(x.permute(0, 2, 1))  # 平均池化  32  512 192
        x = x.permute(0, 2, 1)  # 32 192 512
        return x


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self,use_conv_filters_front, kernel_size, d_model):
        super(series_decomp_multi, self).__init__()
        self.use_conv_filters_front = use_conv_filters_front
        self.kernel_size = kernel_size  # [13,17]
        if self.use_conv_filters_front == 1:
            self.conv_filters = [ConvolutionFilter(kernel, d_model, stride=1) for kernel in kernel_size]
        else:
            self.conv_filters = [moving_avg(kernel, stride=1) for kernel in kernel_size]


    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.conv_filters:  # 池化  k=13和17
            conv_filter = func(x)
            moving_mean.append(conv_filter)
            sea = x - conv_filter  # 32 *96 *7  原值减去平均值
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean  # sea--x减去平均池化   moving_mean平均池化


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self,use_conv_filters_rear, kernel_size, d_model):
        super(series_decomp, self).__init__()
        self.use_conv_filters_rear =use_conv_filters_rear
        if self.use_conv_filters_rear == 1:
            self.moving_avg = ConvolutionFilter_512(kernel_size, d_model, stride=1)
        else:
            self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size * 2)  # 512  2048
        self.relu = nn.ReLU()

        # self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size * 2, filter_size * 2)
        self.layer3 = nn.Linear(filter_size * 2, hidden_size)

        # self.initialize_weight(self.layer1)
        # self.initialize_weight(self.layer2)
        # self.initialize_weight(self.layer3)
        self._initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class BasicInceptionBlock(nn.Module):
    def __init__(self, b1avg_channle, b1_1x1_in_channle, b1_1x1_out, b1_1x1):
        super(BasicInceptionBlock, self).__init__()
        # 2.1 第一层池化 + 1*1卷积  out_size = （in_size - kernel_size + 2padding）/ stride +1
        self.branch1avgPool = nn.AdaptiveAvgPool1d(self.feature_size)
        self.branch1_1x1 = nn.Conv1d(in_channels=self.feature_size,  # 输入通道
                                     out_channels=self.feature_size,  # 输出通道
                                     kernel_size=1, padding=0, stride=1)  # 卷积核大小1*1
        # 2.2 第二层1*1卷积
        self.branch2_1x1 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=1, padding=0, stride=1)

        # 2.3 第三层
        self.branch3_1_1x1 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=1, padding=0, stride=1)
        self.branch3_2_5x5 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=5, padding=0, stride=1)
        # padding=2,因为要保持输出的宽高保持一致

        # 2.4 第四层
        self.branch4_1_1x1 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=1, padding=0, stride=1)
        self.branch4_2_3x3 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=3, padding=0, stride=1)
        self.branch4_3_3x3 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=3, padding=0, stride=1)


class IsometricInceptionBlock(nn.Module):
    def __init__(self, configs, kernel_size=0):
        super(IsometricInceptionBlock, self).__init__()
        self.configs = configs
        self.feature_size = configs.d_model
        self.isometric_kernel = configs.isometric_kernel

        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=self.feature_size, out_channels=self.feature_size,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in self.isometric_kernel])

        # 2.1 第一层池化 + 1*1卷积  out_size = （in_size - kernel_size + 2padding）/ stride +1
        self.branch1avgPool = nn.AdaptiveAvgPool1d(self.feature_size)
        self.branch1_1x1 = nn.Conv1d(in_channels=self.feature_size,  # 输入通道
                                     out_channels=self.feature_size,  # 输出通道
                                     kernel_size=1, padding=0, stride=1)  # 卷积核大小1*1
        # 2.2 第二层1*1卷积
        self.branch2_1x1 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=1, padding=0, stride=1)

        # 2.3 第三层
        self.branch3_1_1x1 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=1, padding=0, stride=1)
        self.branch3_2_5x5 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=5, padding=0, stride=1)
        # padding=2,因为要保持输出的宽高保持一致

        # 2.4 第四层
        self.branch4_1_1x1 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=1, padding=0, stride=1)
        self.branch4_2_3x3 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=3, padding=0, stride=1)
        self.branch4_3_3x3 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=3, padding=0, stride=1)

        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)
        ##合并
        self.merge = nn.Conv2d(self.feature_size, self.feature_size, kernel_size=(4, 1))
        self.act = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_1_avg = self.branch1avgPool(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_1_branch1 = self.branch1_1x1(x_1_avg)
        x_1_branch1 = self.act(x_1_branch1)

        x_2_branch1 = self.branch2_1x1(x)
        x_2_branch1 = self.act(x_2_branch1)

        x_3_zeros = torch.zeros((x.shape[0], x.shape[1], (0 - 1) + 5), device=x.device)
        x_3_branch1 = self.branch3_1_1x1(x)
        x_3_branch1_with_zero = torch.cat((x_3_zeros, x_3_branch1), dim=-1)
        x_3_branch5 = self.branch3_2_5x5(x_3_branch1_with_zero)
        x_3_branch5 = self.act(x_3_branch5)

        x_4_zeros = torch.zeros((x.shape[0], x.shape[1], (0 - 1) + 3), device=x.device)

        x_4_branch1 = self.branch4_1_1x1(x)
        x_4_branch3_with_zero = torch.cat((x_4_branch1, x_4_zeros), dim=-1)
        x_4_branch3 = self.branch4_2_3x3(x_4_branch3_with_zero)
        x_4_branch3_with_zero = torch.cat((x_4_branch3, x_4_zeros), dim=-1)
        x_4_branch3 = self.branch4_3_3x3(x_4_branch3_with_zero)
        x_4_branch3 = self.act(x_4_branch3)

        x_branchs_list = [x_1_branch1, x_2_branch1, x_3_branch5, x_4_branch3]
        merge_list = torch.tensor([], device=x.device)
        for i in range(len(x_branchs_list)):
            x_branchs = x_branchs_list[i].permute(0, 2, 1)
            merge_list = torch.cat((merge_list, x_branchs.unsqueeze(1)), dim=1)
        merge_list = merge_list.permute(0, 3, 1, 2)
        merge_list_result = self.merge(merge_list).squeeze(-2)
        merge_list_result = self.act(merge_list_result)
        return merge_list_result


class DownSamplingInceptionBlock(nn.Module):
    # padding=k//2, stride=k
    def __init__(self, configs):
        super(DownSamplingInceptionBlock, self).__init__()
        self.configs = configs

        self.feature_size = configs.d_model
        # 2.1 第一层池化 + 1*1卷积  out_size = （in_size - kernel_size + 2padding）/ stride +1
        self.branch1avgPool = nn.AdaptiveAvgPool1d(self.feature_size)
        self.branch1_1x1 = nn.Conv1d(in_channels=self.feature_size,  # 输入通道
                                     out_channels=self.feature_size,  # 输出通道
                                     kernel_size=1, padding=0, stride=1)  # 卷积核大小1*1
        # 2.2 第二层1*1卷积
        self.branch2_1x1 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=1, padding=0, stride=1)

        # 2.3 第三层
        self.branch3_1_1x1 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=1, padding=0, stride=1)
        self.branch3_2_5x5 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=5, padding=2, stride=5)
        # padding=2,因为要保持输出的宽高保持一致

        # 2.4 第四层
        self.branch4_1_1x1 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=1, padding=0, stride=1)
        self.branch4_2_3x3 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=3, padding=1, stride=3)
        self.branch4_3_3x3 = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=3, padding=1, stride=3)

        ##合并
        self.merge = nn.Conv2d(self.feature_size, self.feature_size, kernel_size=(4, 1))

        self.act = nn.LeakyReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_1_avg = self.branch1avgPool(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_1_branch1 = self.act(self.branch1_1x1(x_1_avg))

        x_2_branch1 = self.act(self.branch2_1x1(x))

        x_3_zeros = torch.zeros((x.shape[0], x.shape[1], 4 * (x.shape[2] - 1)), device=x.device)
        x_3_branch1 = self.branch3_1_1x1(x)
        x_3_branch1_with_zero = torch.cat((x_3_zeros, x_3_branch1), dim=-1)
        x_3_branch5 = self.act(self.branch3_2_5x5(x_3_branch1_with_zero))

        x_4_zeros = torch.zeros((x.shape[0], x.shape[1], (x.shape[2] - 1) * 2), device=x.device)

        x_4_branch1 = self.branch4_1_1x1(x)
        x_4_branch3_with_zero = torch.cat((x_4_branch1, x_4_zeros), dim=-1)
        x_4_branch3_1 = self.branch4_2_3x3(x_4_branch3_with_zero)
        x_4_branch3_with_zero = torch.cat((x_4_branch3_1, x_4_zeros), dim=-1)
        x_4_branch3_2 = self.act(self.branch4_3_3x3(x_4_branch3_with_zero))

        x_branchs_list = [x_1_branch1, x_2_branch1, x_3_branch5, x_4_branch3_2]
        merge_list = torch.tensor([], device=x.device)
        for i in range(len(x_branchs_list)):
            x_branchs = x_branchs_list[i].permute(0, 2, 1)
            merge_list = torch.cat((merge_list, x_branchs.unsqueeze(1)), dim=1)
        merge_list = merge_list.permute(0, 3, 1, 2)
        merge_list_result = self.merge(merge_list).squeeze(-2)
        merge_list_result = self.act(merge_list_result)

        return merge_list_result


class UpSamplingInceptionBlock(nn.Module):
    def __init__(self, configs):
        super(UpSamplingInceptionBlock, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.feature_size = configs.d_model
        # 2.1 第一层池化 + 1*1卷积  out_size = （in_size - kernel_size + 2padding）/ stride +1
        self.branch1avgPool = nn.AdaptiveAvgPool1d(self.feature_size)
        self.branch1_1x1 = nn.ConvTranspose1d(in_channels=self.feature_size,  # 输入通道
                                              out_channels=self.feature_size,  # 输出通道
                                              kernel_size=1, padding=0, stride=1)  # 卷积核大小1*1
        # 2.2 第二层1*1卷积
        self.branch2_1x1 = nn.ConvTranspose1d(self.feature_size, self.feature_size, kernel_size=1, padding=0, stride=1)

        # 2.3 第三层
        self.branch3_1_1x1 = nn.ConvTranspose1d(self.feature_size, self.feature_size, kernel_size=1, padding=0,
                                                stride=1)
        self.branch3_2_5x5 = nn.ConvTranspose1d(self.feature_size, self.feature_size, kernel_size=5, padding=0,
                                                stride=5)
        self.branch3_2_5x5_linear = nn.Linear((self.pred_len * 2 - 1) * 5 + 5, self.pred_len * 2)
        # padding=2,因为要保持输出的宽高保持一致

        # 2.4 第四层
        self.branch4_1_1x1 = nn.ConvTranspose1d(self.feature_size, self.feature_size, kernel_size=1, padding=0,
                                                stride=1)
        self.branch4_2_3x3 = nn.ConvTranspose1d(self.feature_size, self.feature_size, kernel_size=3, padding=0,
                                                stride=3)
        self.branch4_3_3x3 = nn.ConvTranspose1d(self.feature_size, self.feature_size, kernel_size=3, padding=0,
                                                stride=3)
        self.branch4_3_3x3_linear = nn.Linear((((self.pred_len * 2 - 1) * 3 + 3) - 1) * 3 + 3, self.pred_len * 2)

        ##合并
        self.merge = nn.Conv2d(self.feature_size, self.feature_size, kernel_size=(4, 1))
        self.act = nn.LeakyReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_1_avg = self.branch1avgPool(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_1_branch1 = self.act(self.branch1_1x1(x_1_avg))

        x_2_branch1 = self.act(self.branch2_1x1(x))

        # x_3_zeros = torch.zeros((x.shape[0], x.shape[1], (0 - 1)+1), device=x.device)
        x_3_branch1 = self.branch3_1_1x1(x)
        # x_3_branch1_with_zero = torch.cat((x_3_zeros, x_3_branch1), dim=-1)
        x_3_branch5 = self.branch3_2_5x5(x_3_branch1)
        x_3_branch5 = self.branch3_2_5x5_linear(x_3_branch5)
        x_3_branch5 = self.act(x_3_branch5)
        # x_4_zeros = torch.zeros((x.shape[0], x.shape[1], (x.shape[2] - 1) * 2), device=x.device)

        x_4_branch1 = self.branch4_1_1x1(x)
        # x_4_branch3_with_zero = torch.cat((x_4_branch1, x_4_zeros), dim=-1)
        x_4_branch3 = self.branch4_2_3x3(x_4_branch1)
        # x_4_branch3_with_zero = torch.cat((x_4_branch3, x_4_zeros), dim=-1)
        x_4_branch3 = self.branch4_3_3x3(x_4_branch3)
        x_4_branch3 = self.branch4_3_3x3_linear(x_4_branch3)
        x_4_branch3 = self.act(x_4_branch3)

        x_branchs_list = [x_1_branch1, x_2_branch1, x_3_branch5, x_4_branch3]
        merge_list = torch.tensor([], device=x.device)
        for i in range(len(x_branchs_list)):
            x_branchs = x_branchs_list[i].permute(0, 2, 1)
            merge_list = torch.cat((merge_list, x_branchs.unsqueeze(1)), dim=1)
        merge_list = merge_list.permute(0, 3, 1, 2)
        merge_list_result = self.merge(merge_list).squeeze(-2)
        merge_list_result = self.act(merge_list_result)
        return merge_list_result


class MultiConvBlock(nn.Module):
    def __init__(self, configs):
        super(MultiConvBlock, self).__init__()
        self.configs = configs
        self.use_conv_filters_rear = configs.use_conv_filters_rear
        self.feature_size = configs.d_model
        self.d_model = configs.d_model
        # self.feature_size = configs.feature_size
        self.isometric_kernel = configs.isometric_kernel
        self.conv_kernel = configs.conv_kernel
        self.decomp_kernel = configs.decomp_kernel
        self.use_season_fourier_decom_model_rear = configs.use_season_fourier_decom_model_rear
        self.use_fourier_embed_pred_rear = configs.use_fourier_embed_pred_rear
        self.use_season_attention_rear = configs.use_season_attention_rear

        self.isometricInceptionBlocks = nn.Sequential(
            IsometricInceptionBlock(self.configs),

            nn.GELU(),
            IsometricInceptionBlock(self.configs),
        )

        self.downSamplingInceptionBlocks = nn.Sequential(
            DownSamplingInceptionBlock(self.configs),
            nn.GELU(),
            DownSamplingInceptionBlock(self.configs),

        )

        self.upSamplingInceptionBlocks = nn.Sequential(
            UpSamplingInceptionBlock(self.configs),
            nn.GELU(),
            UpSamplingInceptionBlock(self.configs),

        )
        if self.use_season_fourier_decom_model_rear == 0:
            self.decomp = nn.ModuleList([series_decomp(self.use_conv_filters_rear,k, self.d_model) for k in self.decomp_kernel])

        if self.use_season_attention_rear == 1:
            self.season_attention_rear = AttentionLayer(FullAttention(), d_model=7, n_heads=7)

        if self.use_fourier_embed_pred_rear == 1:
            self.fourier_embed_rear = FourierEmbeddingPredictor(configs)
        if self.use_season_fourier_decom_model_rear == 1:
            self.decomp = nn.ModuleList([FourierDecmLayer(pred_len=0, k=2) for _ in range(len(self.decomp_kernel))])


        self.merge = torch.nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))

        self.fnn = FeedForwardNetwork(self.feature_size, self.feature_size * 4, dropout_rate=0.05)  # 前馈神经网络
        self.fnn_norm = torch.nn.LayerNorm(self.feature_size)

        self.norm = torch.nn.LayerNorm(self.feature_size)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.05)
        # self.testModel = Simam_module()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, src):
        if self.use_fourier_embed_pred_rear == 1:
            src = self.fourier_embed_rear(src)
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, _ = self.decomp[i](src)

            if self.use_season_attention_rear == 1:
                atten_dec_out, _ = self.season_attention_rear(src_out, src_out, src_out, attn_mask=None)
                src_out = src_out + atten_dec_out



            x = src_out.permute(0, 2, 1)

            x_downSamplingInceptionBlocks = self.downSamplingInceptionBlocks(
                x)  # x=32*7*192  x_downSamplingInceptionBlocks 32*7*192
            x_downSamplingInceptionBlocks = self.act(x_downSamplingInceptionBlocks)

            x_isometricInceptionBlocks = self.isometricInceptionBlocks(x_downSamplingInceptionBlocks)
            x_isometricInceptionBlocks = self.act(x_isometricInceptionBlocks)


            multi.append(x_isometricInceptionBlocks)

            # merge
        mg = torch.tensor([], device=src.device)
        for i in range(len(self.conv_kernel)):
            x_branchs = multi[i].permute(0, 2, 1)

            mg = torch.cat((mg, x_branchs.unsqueeze(1)), dim=1)  # 32 2 512 192
        mg = mg.permute(0, 3, 1, 2)
        mg = self.merge(mg).squeeze(-2).permute(0, 2, 1)  # 32 192 512

        return self.fnn_norm(mg + self.fnn(mg))




class MultiModelBlock(nn.Module):

    def __init__(self, configs):
        super(MultiModelBlock, self).__init__()
        # self.embed_dist = {"exchange_rate.csv":21,"weather_old.csv":8,"ETTh1.csv":7,"ETTh2.csv":7,"ETTm1.csv":7,"ETTm2.csv":7}
        self.configs = configs
        self.use_conv_filters_front = configs.use_conv_filters_front
        self.use_season_fourier_decom_model_front = configs.use_season_fourier_decom_model_front
        self.use_season_attention_front = configs.use_season_attention_front
        self.use_trend_attention = configs.use_trend_attention
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len  # 96
        self.seq_len = configs.seq_len  # 96
        self.c_out = configs.c_out  # 7
        self.d_model = configs.d_model
        self.multi_layers = configs.multi_layers
        self.decomp_kernel = configs.decomp_kernel  # [13,17]
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        self.decomp_multi = series_decomp_multi(self.use_conv_filters_front,self.decomp_kernel, self.d_model)

        self.up_predlen_layer = nn.Linear(self.seq_len, self.pred_len)

        ################## 傅里叶分解season
        if self.use_season_fourier_decom_model_front == 1:
            self.season_fourier_decom_model_front = FourierDecmLayer(pred_len=0, k=2)

        if self.use_trend_attention == 1:
            self.trend_attention =AttentionLayer(FlowAttention(),d_model=7,n_heads=7)

        self.regression = nn.Linear(self.pred_len, self.pred_len)
        # self.regression = nn.Linear(self.pred_len, self.pred_len)


        self.multi_conv_lavers = nn.ModuleList([MultiConvBlock(configs=configs)
                                                for _ in range(self.multi_layers)])
        self.seasonal_linear = nn.Linear(self.c_out, self.d_model)


        self.x_mark_dec_linear = nn.Linear(4, self.d_model)
        # self.x_mark_dec_linear = nn.Linear(self.embed_dist[self.configs.data_path], self.d_model)
        # self.trend_out_linear = nn.Linear(self.pred_len,  self.pred_len * 2)
        # 将384改成288
        self.trend_out_linear_1 = nn.Linear(self.pred_len, self.pred_len + self.label_len)
        self.relu = nn.ReLU()  # 激活函数

        if self.use_season_attention_front == 1:
            self.season_attention_front = AttentionLayer(FullAttention(),d_model=self.d_model,n_heads=self.d_model)



    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):

        x = self.up_predlen_layer(batch_x.permute(0, 2, 1)).permute(0, 2, 1)


        seasonal_init_enc, trend = self.decomp_multi( x)  # 序列特征分解



        if self.use_season_fourier_decom_model_front == 1:
            seasonal_init_enc, _ = self.season_fourier_decom_model_front( x)


        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.trend_out_linear_1(trend.permute(0, 2, 1)).permute(0, 2, 1)
        if self.use_trend_attention == 1:
            trend_att = trend
            out, attn = self.trend_attention(trend_att, trend_att,trend_att,attn_mask=None)
            trend = trend + out

        # embedding
        zeros = torch.zeros([dec_inp.shape[0], self.label_len, dec_inp.shape[2]], device=dec_inp.device)
        # zeros = torch.zeros([dec_inp.shape[0], self.seq_len, dec_inp.shape[2]], device=dec_inp.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.pred_len:, :], zeros], dim=1)


        # ++++++++++++++++++++++++++++++++++++ 改

        seasonal_init_dec = self.seasonal_linear(seasonal_init_dec)

        batch_y_mark = self.x_mark_dec_linear(batch_y_mark)
        dec_out = seasonal_init_dec + batch_y_mark

        if self.use_season_attention_front == 1:
            atten_dec_out,_ = self.season_attention_front(dec_out,dec_out,dec_out,attn_mask=None)
            dec_out = dec_out + atten_dec_out

        for i in range(self.multi_layers):

            dec_out = self.multi_conv_lavers[i](dec_out)

        dec_out = dec_out + trend

        return dec_out


class FGN(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.pred_len = configs.pred_len  # 改  # 预测长度  12  --- 改成96
        self.seq_len = configs.seq_len  # 12 输入序列的一条数据的长度 或者维度  ---- 改成96
        self.d_model = configs.d_model
        self.use_fourier_embed_pred_front = configs.use_fourier_embed_pred_front
        # FourierBlock
        if self.use_fourier_embed_pred_front == 1:
            self.fourier_embed_pred = FourierEmbeddingPredictor(configs)

        # MultiModelBlock
        self.multi_model_block = MultiModelBlock(configs)

        # merge
        self.f_merge_multi = nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=(2, 1),
                                       stride=1)
        self.to('cuda:0')


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # FourierGNN

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        x = batch_x

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev  # 32*96*7

        # ------------------------------

        # FourierBlock
        if self.use_fourier_embed_pred_front == 1:
            x_i = self.fourier_embed_pred(x)  # 结果是32*192*7
        else:
            x_i = x


        x_multi = self.multi_model_block(x_i, batch_x_mark, dec_inp, batch_y_mark)


        x_f_x_multi_result_merge = x_multi
        x_f_x_multi_result_merge = x_f_x_multi_result_merge * \
                                   (stdev[:, 0, :].unsqueeze(1).repeat(
                                       1, self.pred_len + self.seq_len, 1))
        x_f_x_multi_result_merge = x_f_x_multi_result_merge + \
                                   (means[:, 0, :].unsqueeze(1).repeat(
                                       1, self.pred_len + self.seq_len, 1))
        return x_f_x_multi_result_merge


##FourierEmbeddingPredictor  SpectralProjectionNetwork
class FourierEmbeddingPredictor(nn.Module):
    def __init__(self, configs):
        super(FourierEmbeddingPredictor, self).__init__()
        self.embed_size = configs.embed_size  # 128  一个batch包含的样本个数
        self.seq_len = configs.seq_len  # 输入序列的一条数据的长度 或者维度  ---- 改成96
        self.number_frequency = 1

        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_len, 8))  # 12*8
        self.hidden_size = configs.hidden_size
        self.pred_len = configs.pred_len

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),  # 1024*64
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size * 2),  # 64*256
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size * 2, self.pred_len)  # 256*12
        )



        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))  # 1*128


        # self.projection = nn.Linear(
        #     configs.d_model, configs.c_out, bias=True)

        self.predict_linear = nn.Linear(
            self.pred_len, self.seq_len)

        self.projection = nn.Linear(
            configs.c_out, configs.d_model, bias=True)

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()  # 32*7*96

        B, N, L = x.shape

        # B*N*L ==> B*NL
        x = x.reshape(B, -1)  # 32*672

        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)  # 32*672*128

        # FFT B*NL*D ==> B*NT/2*D
        x_f = torch.fft.rfft(x, dim=1, norm='ortho')  # 32*337*128   672/2+1

        x_f = x_f.reshape(B, (N * L) // 2 + 1, self.embed_size)  # 32*337*128

        bias = x_f

        x_f = x_f + bias  # 32*337*128  残差

        x_f = x_f.reshape(B, (N * L) // 2 + 1, self.embed_size)  # 32*337*128

        # ifft

        x_i = torch.fft.irfft(x_f, n=N * L, dim=1, norm="ortho")  # 32*672*128

        x_i = x_i.reshape(B, N, L, self.embed_size)  # 32 *7*96*128
        x_i = x_i.permute(0, 1, 3, 2)  # B, N, D, L 32*7*128*96
        ##############################################################################

        # projection

        x_i = torch.matmul(x_i, self.embeddings_10)  # 32*7*128*96 * 96*8  = 32*7*128*8
        '''
            原：x_i = 32*7*128*96
        '''

        x_i = x_i.reshape(B, N, -1)  # 32*7*1024
        x_i = self.fc(x_i)

        x_i = x_i.permute(0, 2, 1).contiguous()  # 32 * 96 *7

        x_i = self.predict_linear(x_i.permute(0, 2, 1)).permute(0, 2, 1)  # 7*192*16

        x_i = self.projection(x_i)



        return x_i




class FourierDecmLayer(nn.Module):

    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        # self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""

        if self.output_attention:
            return self.dft_forward(x)

        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        T = x.size(1)

        dft_mat = torch.fft.fft(torch.eye(T))
        i, j = torch.meshgrid(torch.arange(self.pred_len + T), torch.arange(T))
        omega = np.exp(2 * math.pi * 1j / T)
        idft_mat = (np.power(omega, i * j) / T).cfloat()

        x_freq = torch.einsum('ft,btd->bfd', [dft_mat, x.cfloat()])

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:T // 2]
        else:
            x_freq = x_freq[:, self.low_freq:T // 2 + 1]

        _, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        dft_mat = repeat(dft_mat, 'f t -> b f t d', b=x.shape[0], d=x.shape[-1])
        idft_mat = repeat(idft_mat, 't f -> b t f d', b=x.shape[0], d=x.shape[-1])

        mesh_a, mesh_b = torch.meshgrid(torch.arange(x.size(0)), torch.arange(x.size(2)))

        dft_mask = torch.zeros_like(dft_mat)
        dft_mask[mesh_a, indices, :, mesh_b] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat)
        idft_mask[mesh_a, :, indices, mesh_b] = 1
        idft_mat = idft_mat * idft_mask

        attn = torch.einsum('bofd,bftd->botd', [idft_mat, dft_mat]).real
        return torch.einsum('botd,btd->bod', [attn, x]), rearrange(attn, 'b o t d -> b d o t')