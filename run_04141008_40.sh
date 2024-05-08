#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH -w aiwkr1


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
seq_len=96
if [ ! -d "./log_04141008_40" ]; then
    mkdir ./log_04141008_40
fi

if [ ! -d "./log_04141008_40/ETTm1" ]; then
    mkdir ./log_04141008_40/ETTm1
fi
if [ ! -d "./log_04141008_40/ETTh1" ]; then
    mkdir ./log_04141008_40/ETTh1
fi
if [ ! -d "./log_04141008_40/ETTm2" ]; then
    mkdir ./log_04141008_40/ETTm2
fi

if [ ! -d "./log_04141008_40/ETTh2" ]; then
    mkdir ./log_04141008_40/ETTh2
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


#  m2 2分解核

#seq_len=96
#e_layer=2
#down_sampling_layers=3
#down_sampling_window=2
#learning_rate=0.01
#m2_d_model=32
m2_d_model=128
#m2_d_ff=2048
m2_d_ff=32
m2_batch_size=16
decomp_kernel=(17 49 81)
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --root_path ETT-small \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --d_layers 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 512 \
        --d_ff 16 \
        --batch_size $m2_batch_size \
        --decomp_kernel $decomp_kernel \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm2/'d_ff= 16'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --root_path ETT-small \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --d_layers 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 512 \
        --d_ff 32 \
        --batch_size $m2_batch_size \
        --decomp_kernel $decomp_kernel \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm2/'d_ff= 32'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --root_path ETT-small \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --d_layers 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 512 \
        --d_ff 64 \
        --batch_size $m2_batch_size \
        --decomp_kernel $decomp_kernel \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm2/'d_ff= 64'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --root_path ETT-small \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --d_layers 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 512 \
        --d_ff 128 \
        --batch_size $m2_batch_size \
        --decomp_kernel $decomp_kernel \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm2/'d_ff= 128'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --root_path ETT-small \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --d_layers 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 512 \
        --d_ff 256 \
        --batch_size $m2_batch_size \
        --decomp_kernel $decomp_kernel \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm2/'d_ff= 256'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --root_path ETT-small \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --d_layers 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 512 \
        --d_ff 512 \
        --batch_size $m2_batch_size \
        --decomp_kernel $decomp_kernel \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm2/'d_ff= 512'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --root_path ETT-small \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --d_layers 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 512 \
        --d_ff 1024 \
        --batch_size $m2_batch_size \
        --decomp_kernel $decomp_kernel \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm2/'d_ff= 1024'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --root_path ETT-small \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --d_layers 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 512 \
        --d_ff 2048 \
        --batch_size $m2_batch_size \
        --decomp_kernel $decomp_kernel \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm2/'d_ff= 2048'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done






# m1  learing_rate=0.01  dmodel=16    batch_size=16

#seq_len=96
#e_layer=2
#down_sampling_layers=3
#down_sampling_window=2
#learning_rate=0.01
#m1_d_model=16
m1_d_model=128
#m1_d_ff=2048
m1_d_ff=32
m1_batch_size=16
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 1024 \
        --d_ff 16 \
        --batch_size $m1_batch_size \
        --decomp_kernel $decomp_kernel \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm1/'d_ff= 16'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 1024 \
        --d_ff 32 \
        --batch_size $m1_batch_size \
        --decomp_kernel $decomp_kernel \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm1/'d_ff= 32'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 1024 \
        --d_ff 64 \
        --batch_size $m1_batch_size \
        --decomp_kernel $decomp_kernel \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm1/'d_ff= 64'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 1024 \
        --d_ff 128 \
        --batch_size $m1_batch_size \
        --decomp_kernel $decomp_kernel \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm1/'d_ff= 128'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 1024 \
        --d_ff 256 \
        --batch_size $m1_batch_size \
        --decomp_kernel $decomp_kernel \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm1/'d_ff= 256'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 1024 \
        --d_ff 512 \
        --batch_size $m1_batch_size \
        --decomp_kernel $decomp_kernel \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm1/'d_ff= 512'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 1024 \
        --d_ff 1024 \
        --batch_size $m1_batch_size \
        --decomp_kernel $decomp_kernel \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm1/'d_ff= 1024'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 1024 \
        --d_ff 2048 \
        --batch_size $m1_batch_size \
        --decomp_kernel $decomp_kernel \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04141008_40/ETTm1/'d_ff= 2048'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done








