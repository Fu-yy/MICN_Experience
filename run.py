import argparse
import torch
import numpy as np
import random
from exp.exp_informer import Exp_Informer



if __name__ == '__main__':

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(2021)


    parser = argparse.ArgumentParser(description='[MICN] Long Sequences Forecasting')

    parser.add_argument('--model', type=str, required=True, default='micn',help='model of experiment: MICN')
    parser.add_argument('--mode', type=str, default='regre', help='different mode of trend prediction block: [regre or mean]')

    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--conv_kernel', type=int, nargs='+', default=[17,49], help='downsampling and upsampling convolution kernel_size')
    parser.add_argument('--decomp_kernel', type=int, nargs='+', default=[17,49], help='decomposition kernel_size')
    parser.add_argument('--isometric_kernel', type=int, nargs='+', default=[17,49], help='isometric convolution kernel_size')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--output_attention',action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')

    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--comment', type=str, default='none', help='com')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

    # 更改 2024.4.1 添加invert——embed
    parser.add_argument('--use_invertembed', type=int, default=1,help='use_invertembed')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    # 更改 2024.4.1 添加x_enc  x_mark 最后一个维度

    parser.add_argument('--x_enc_len', type=int, default=7, help='x_enc_len')
    parser.add_argument('--x_mark_len', type=int, default=5, help='x_mark_len')
    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')


    # TimeMixer

    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')


    # 消融
    parser.add_argument('--use_fourier', type=int, default=1,help='use_fourier')
    parser.add_argument('--use_space_merge', type=int, default=1,help='use_space_merge')
    parser.add_argument('--pred_use_conv', type=int, default=1,help='pred_use_conv')
    parser.add_argument('--season_use_fourier', type=int, default=1,help='season_use_forier')
    parser.add_argument('--trend_use_conv', type=int, default=1,help='trend_use_conv')
    parser.add_argument('--cut_freq', type=int, default=10,help='cut_freq')


    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'WTH':{'data':'weather.csv','T':'OT','M':[21,21,21],'S':[1,1,1],'MS':[21,21,1]},
        'ECL':{'data':'electricity.csv','T':'OT','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
        'Traffic': {'data': 'traffic.csv', 'T': 'OT', 'M': [862, 862, 862], 'S': [1, 1, 1], 'MS': [862, 862, 1]},
        'Exchange': {'data': 'exchange_rate.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1]},
        'ILI': {'data': 'national_illness.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'PEMS03': {'data': 'PEMS03.npz', 'T': 'OT', 'M': [358, 358, 358], 'S': [1, 1, 1], 'MS': [358, 358, 1]},
        'PEMS04': {'data': 'PEMS04.npz', 'T': 'OT', 'M': [307, 307, 307], 'S': [1, 1, 1], 'MS': [307, 307, 1]},
        'PEMS07': {'data': 'PEMS07.npz', 'T': 'OT', 'M': [883, 883, 883], 'S': [1, 1, 1], 'MS': [883, 883, 1]},
        'PEMS08': {'data': 'PEMS08.npz', 'T': 'OT', 'M': [170, 170, 170], 'S': [1, 1, 1], 'MS': [170, 170, 1]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    decomp_kernel = []  # kernel of decomposition operation
    isometric_kernel = []  # kernel of isometric convolution
    for ii in args.conv_kernel:
        if ii%2 == 0:   # the kernel of decomposition operation must be odd
            decomp_kernel.append(ii+1)
            isometric_kernel.append((args.seq_len + args.pred_len+ii) // ii)
        else:
            decomp_kernel.append(ii)
            isometric_kernel.append((args.seq_len + args.pred_len+ii-1) // ii)
    args.isometric_kernel = isometric_kernel  # kernel of isometric convolution
    args.decomp_kernel = decomp_kernel   # kernel of decomposition operation


    print('Args in experiment:')
    print(args)
    Exp = Exp_Informer
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_{}'.format(args.model, args.data, args.mode, args.features,
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                    args.embed, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
