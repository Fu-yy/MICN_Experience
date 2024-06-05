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
# 新      v202405261243       0 1 1 1  加上消融                          2024.5.29 19:23
# 新      v202405261243       1 0 1 1  加上消融                          2024.5.29 20:00
# 新      v202405261243       1 1 0 1  加上消融                          2024.5.29 20:10
# 新      v202405261243       1 1 1 0  加上消融                          2024.5.29 20:20
# 新      v202405291900       0 0 0 1                                    2024.6.1 8:56
# 新      v202405291900       0 1 0 1                                    2024.6.1 9:09
# 新      v202405291900       0 1 1 0                                    2024.6.1 9:09
# 新      v202405291900       0 1 0 0                                    2024.6.1 20：43
# 新      v202405291900       0 0 1 0                                    2024.6.3 15：26


use_x_mark_enc=0
front_use_decomp=0
use_fourier=1
use_space_merge=0


seq_len=96
if [ ! -d "./log_06031526_40" ]; then
    mkdir ./log_06031526_40
fi

if [ ! -d "./log_06031526_40/ETTm1" ]; then
    mkdir ./log_06031526_40/ETTm1
fi
if [ ! -d "./log_06031526_40/ETTh1" ]; then
    mkdir ./log_06031526_40/ETTh1
fi
if [ ! -d "./log_06031526_40/ETTm2" ]; then
    mkdir ./log_06031526_40/ETTm2
fi

if [ ! -d "./log_06031526_40/ETTh2" ]; then
    mkdir ./log_06031526_40/ETTh2
fi
if [ ! -d "./log_06031526_40/electricity" ]; then
    mkdir ./log_06031526_40/electricity
fi

if [ ! -d "./log_06031526_40/Exchange" ]; then
    mkdir ./log_06031526_40/Exchange
fi

if [ ! -d "./log_06031526_40/Solar" ]; then
    mkdir ./log_06031526_40/Solar
fi

if [ ! -d "./log_06031526_40/weather" ]; then
    mkdir ./log_06031526_40/weather
fi

if [ ! -d "./log_06031526_40/Traffic" ]; then
    mkdir ./log_06031526_40/Traffic
fi

if [ ! -d "./log_06031526_40/PEMS03" ]; then
    mkdir ./log_06031526_40/PEMS03
fi

if [ ! -d "./log_06031526_40/PEMS04" ]; then
    mkdir ./log_06031526_40/PEMS04
fi

if [ ! -d "./log_06031526_40/PEMS07" ]; then
    mkdir ./log_06031526_40/PEMS07
fi
if [ ! -d "./log_06031526_40/PEMS08" ]; then
    mkdir ./log_06031526_40/PEMS08
fi
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

Solar_x_mark_len=1
PEMS03_x_mark_len=1
PEMS04_x_mark_len=1
PEMS07_x_mark_len=1
PEMS08_x_mark_len=1



#  h1 2分解核  dmodel=dff=256

for pred_len in 96  192 336 720; do
 python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
      --model micn \
      --root_path ETT-small \
      --mode regre \
      --data ETTh1 \
      --freq h \
      --features M \
      --e_layers 2 \
      --d_layers 1 \
      --d_model 256 \
      --d_ff 256 \
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
      --front_use_decomp $front_use_decomp \
      --use_x_mark_enc $use_x_mark_enc \
  > log_06031526_40/ETTh1/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done





#  h2 2分解核  d_model= d_ff = 128

for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTh2 \
        --freq h \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
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
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/ETTh2/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
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
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/ETTm1/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done



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
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/ETTm2/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done













# electricity d_model=d_ff=512

 for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path electricity \
        --mode regre \
        --data ECL \
        --freq h \
        --features M \
        --e_layers 3 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 512 \
        --batch_size 16 \
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
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/electricity/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done







if false;then
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
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/Traffic/'0'_$model_name'_'Traffic'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

fi



# Exchange d_model=d_ff=128
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path exchange_rate \
        --mode regre \
        --data Exchange \
        --features M \
        --e_layers 2 \
        --freq d \
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
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/Exchange/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
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
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/weather/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done





# Solar
for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path Solar \
        --mode regre \
        --data Solar \
        --freq t \
        --features M \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --learning_rate 0.0005 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $Solar_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/Solar/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done


#PEMS03
for pred_len in 12 24 48 96 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path PEMS \
        --data PEMS03 \
        --freq t \
        --features M \
        --d_layers 1 \
        --e_layers 4 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --learning_rate 0.001 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS03_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/PEMS03/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done


#PEMS04
for pred_len in 12 24 48 96 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path PEMS \
        --data PEMS04 \
        --freq t \
        --features M \
        --d_layers 1 \
        --e_layers 4 \
        --d_ff 1024 \
        --d_model 1024 \
        --seq_len 96 \
        --pred_len $pred_len \
        --learning_rate 0.0005 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS07_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/PEMS04/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
#PEMS07
for pred_len in 12 24 48 96 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path PEMS \
        --data PEMS07 \
        --freq t \
        --features M \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --learning_rate 0.001 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS07_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/PEMS07/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
#PEMS08
for pred_len in 12 24 48 96 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path PEMS \
        --data PEMS08 \
        --freq t \
        --features M \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS08_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06031526_40/PEMS08/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
