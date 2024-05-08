import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomFormer(nn.Module):
    def __init__(self,):
        super(DecomFormer, self).__init__()
        self.branch1avgPool = nn.AdaptiveAvgPool1d(self.feature_size)


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        '''
        MultiScaleSeasonMixing(
          (down_sampling_layers): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=96, out_features=48, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=48, out_features=48, bias=True)
            )
            (1): Sequential(
              (0): Linear(in_features=48, out_features=24, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=24, out_features=24, bias=True)
            )
            (2): Sequential(
              (0): Linear(in_features=24, out_features=12, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=12, out_features=12, bias=True)
            )
          )
        )
        
        '''
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

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


if __name__ == '__main__':
    # a = [1,2,3,4,5,6,7,8,9]
    # parser = argparse.ArgumentParser(description='[MICN] Long Sequences Forecasting')
    # parser.add_argument('--seq_len', type=int, default=96, help='seq_len')
    # parser.add_argument('--down_sampling_window', type=int, default=2, help='seq_len')
    # parser.add_argument('--down_sampling_layers', type=int, default=3, help='seq_len')
    #
    # args = parser.parse_args()
    #
    # testmodel = MultiScaleSeasonMixing(args)
    #
    # x = testmodel(a)
    local_out = torch.randn(32,96,7)
    local_out = F.interpolate(local_out, size=20, mode='linear', align_corners=False)
    c = local_out
    print(local_out.shape)
    # for i in a:
    #     test = a[i]

    # conv_kernel = [12,16]
    # conv_kernel2 = [18,12]
    #
    # decomp_kernel = []
    # for ii in conv_kernel2:
    #     if ii%2 == 0:   # the kernel of decomposition operation must be odd
    #         decomp_kernel.append(ii+1)
    #     else:
    #         decomp_kernel.append(ii)
    # decomp_kernel = decomp_kernel   # kernel of decomposition operation
