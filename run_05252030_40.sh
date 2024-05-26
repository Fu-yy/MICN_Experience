#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH -w aiwkr3


#module load cuda/11.7.0
#module load singularity/3.11.0
module load cuda/11.8.0
module load anaconda/anaconda3-2022.10

source activate py310t2cu118

# 对out_cross_layer_test进行第二维度全连接  取消第三维度全连接     效果在ettm2最好    2024.4.4 14:52
# 添加第三维度全连接 同时将dff设置成2048                       不行             2024.4.4 15：10
# 卷积代替全连接 替换seasonMixing 同时dff-4096                不行             2024.4.4 15：10

# 新增其他数据集                                                               2024.4.4 19：24
# 新增其他数据集     把原来的标准化加回来                                        2024.4.5 09：53
# 更改分解核尺寸、batchsize  dmodel   dff大小                                        2024.4.6 09：28
# 测试d_model 的影响                                                             2024.4.8 20：25
# 测试d_ff 的影响                                                             2024.4.8 20：32
# 替换下采样为DCT下采样                                                             2024.4.10 15：42
# 替换下采样为傅里叶下采样   modle128  ff 32                                          2024.4.11 13：02
# 测试dmodel                                           2024.4.12 17：25
# 测试dff                                           2024.4.14 10:08
# 消融实验 使用itransformer的参数    use_fourier + use_space_merge                                        2024.5.1 14:41
# 消融实验 使用itransformer的参数    use_fourier 0 + use_space_merge                                        2024.5.1 14:42
# 消融实验 使用itransformer的参数    use_fourier 0 + use_space_merge 0                                        2024.5.1 14:43
# 消融实验 使用itransformer的参数    use_fourier 1 + use_space_merge 0                                        2024.5.1 14:44
# 新                                                                                           2024.5.7 21：40
# 新    ett、ELECTRICT                                                                                         2024.5.11 16：04
# 新   傅里叶0 conv2d 1                                                                                       2024.5.12 21:05
# 新   傅里叶0 conv2d 0                                                                                       2024.5.12 21:06
# 新   傅里叶1 conv2d 0                                                                                       2024.5.12 21:07
# 新   use_fourier=0   use_space_merge=1   pred_use_conv=1  season_use_fourier=1  trend_use_conv=1           2024.5.13 16:28
# 新   use_fourier=0   use_space_merge=0   pred_use_conv=1  season_use_fourier=1  trend_use_conv=1           2024.5.13 16:37
# 新   use_fourier=1   use_space_merge=0   pred_use_conv=1  season_use_fourier=1  trend_use_conv=1           2024.5.13 16:38
# 新   use_fourier=1   use_space_merge=1   pred_use_conv=0  season_use_fourier=1  trend_use_conv=1           2024.5.13 16:39
# 新   use_fourier=1   use_space_merge=1   pred_use_conv=1  season_use_fourier=0  trend_use_conv=1           2024.5.13 16:40
# 新   use_fourier=1   use_space_merge=1   pred_use_conv=1  season_use_fourier=1  trend_use_conv=0           2024.5.13 16:41
# 新      v202405252030       Traffic和ECL都用128，batchsize=32          2024.5.25 20:30




seq_len=96
if [ ! -d "./log_05252030_40" ]; then
    mkdir ./log_05252030_40
fi

if [ ! -d "./log_05252030_40/ETTm1" ]; then
    mkdir ./log_05252030_40/ETTm1
fi
if [ ! -d "./log_05252030_40/ETTh1" ]; then
    mkdir ./log_05252030_40/ETTh1
fi
if [ ! -d "./log_05252030_40/ETTm2" ]; then
    mkdir ./log_05252030_40/ETTm2
fi

if [ ! -d "./log_05252030_40/ETTh2" ]; then
    mkdir ./log_05252030_40/ETTh2
fi
if [ ! -d "./log_05252030_40/electricity" ]; then
    mkdir ./log_05252030_40/electricity
fi

if [ ! -d "./log_05252030_40/Exchange" ]; then
    mkdir ./log_05252030_40/Exchange
fi

if [ ! -d "./log_05252030_40/illness" ]; then
    mkdir ./log_05252030_40/illness
fi

if [ ! -d "./log_05252030_40/weather" ]; then
    mkdir ./log_05252030_40/weather
fi

if [ ! -d "./log_05252030_40/Traffic" ]; then
    mkdir ./log_05252030_40/Traffic
fi


use_fourier=1
use_space_merge=1
pred_use_conv=0
season_use_fourier=1
trend_use_conv=0

x
#singularity exec --nv /mnt/nfs/data/home/1120231440/home/fuy/fuypycharm1_1.sif  nvidia-smi;\

date_ETTh1=ETTh1
date_ETTh2=ETTh2
date_ETTm1=ETTm1
date_ETTm2=ETTm2
date_exchange_rate=exchange_rate
date_national_illness=national_illness
date_weather=weather
# 分解核除了ill是 19 13，其他都是13 17  且必须为奇数
decomp_kernel=(17 49 81)
electricity_x_mark_len=4
weather_x_mark_len=5
Exchange_x_mark_len=3
illness_x_mark_len=3
traffic_x_mark_len=4
ETTh1_x_mark_len=4
ETTh2_x_mark_len=4
ETTm2_x_mark_len=5
ETTm1_x_mark_len=5

ETTm2_cut_freq=10
ETTm1_cut_freq=45
ETTh1_cut_freq=10
ETTh2_cut_freq=30


#  m2 2分解核

# m2 d_model= d_ff = 128
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --root_path ETT-small \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --d_layers 1 \
        --freq t \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 128 \
        --d_ff 128 \
        --decomp_kernel $decomp_kernel \
        --x_enc_len 7 \
        --x_mark_len $ETTm2_x_mark_len \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --pred_use_conv $pred_use_conv \
        --season_use_fourier $season_use_fourier \
        --trend_use_conv $trend_use_conv \
        --cut_freq $ETTm2_cut_freq \
    > log_05252030_40/ETTm2/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done




# m1 d_model= d_ff = 128

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --freq t \
        --data ETTm1 \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --decomp_kernel $decomp_kernel \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len $ETTm1_x_mark_len \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --pred_use_conv $pred_use_conv \
        --season_use_fourier $season_use_fourier \
        --trend_use_conv $trend_use_conv \
        --cut_freq $ETTm1_cut_freq \
    > log_05252030_40/ETTm1/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done




#  h1 2分解核  dmodel=dff=256

for pred_len in 96  192 336 720; do
 python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
      --model micn \
      --root_path ETT-small \
      --mode regre \
      --data ETTh1 \
      --freq t \
      --features M \
      --e_layers 2 \
      --d_layers 1 \
      --d_model 256 \
      --d_ff 256 \
      --decomp_kernel $decomp_kernel \
      --itr 1 \
      --x_enc_len 7 \
      --x_mark_len $ETTh1_x_mark_len \
      --seq_len 96 \
      --pred_len $pred_len \
      --down_sampling_layers 3 \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --channel_independence 1 \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --pred_use_conv $pred_use_conv \
      --season_use_fourier $season_use_fourier \
      --cut_freq $ETTh1_cut_freq \
      --trend_use_conv $trend_use_conv \
  > log_05252030_40/ETTh1/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done




#  h2 2分解核  d_model= d_ff = 128

for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTh2 \
        --freq t \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --decomp_kernel $decomp_kernel \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len $ETTh2_x_mark_len \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --pred_use_conv $pred_use_conv \
        --season_use_fourier $season_use_fourier \
        --cut_freq $ETTh1_cut_freq \
        --trend_use_conv $trend_use_conv \
    > log_05252030_40/ETTh2/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done






# electricity d_model=d_ff=512

 for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path electricity \
        --mode regre \
        --freq h \
        --data ECL \
        --features M \
        --e_layers 3 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --learning_rate 0.0005 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --pred_use_conv $pred_use_conv \
        --season_use_fourier $season_use_fourier \
        --trend_use_conv $trend_use_conv \
    > log_05252030_40/electricity/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done








# Traffic d_model=d_ff=512

for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path traffic \
        --mode regre \
        --data Traffic \
        --features M \
        --e_layers 4 \
        --d_layers 1 \
        --freq h \
        --seq_len 96 \
        --itr 1 \
        --factor 3 \
        --d_model 128 \
        --d_ff 128 \
        --learning_rate 0.001 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $traffic_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --pred_use_conv $pred_use_conv \
        --season_use_fourier $season_use_fourier \
        --trend_use_conv $trend_use_conv \
    > log_05252030_40/Traffic/'0'_$model_name'_'Traffic'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done





# Exchange d_model=d_ff=128
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path exchange_rate \
        --mode regre \
        --data Exchange \
        --features M \
        --e_layers 2 \
        --freq t \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --seq_len 96 \
        --itr 1 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $Exchange_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --pred_use_conv $pred_use_conv \
        --season_use_fourier $season_use_fourier \
        --trend_use_conv $trend_use_conv \
    > log_05252030_40/Exchange/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done











#  weather 3分解核



# weather d_ff=d_ff=512
  for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path weather \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --d_layers 1 \
        --e_layers 3 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $weather_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --pred_use_conv $pred_use_conv \
        --season_use_fourier $season_use_fourier \
        --trend_use_conv $trend_use_conv \
    > log_05252030_40/weather/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done





