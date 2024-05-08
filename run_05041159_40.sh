#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH -w aiwkr1


#module load cuda/11.7.0
#module load singularity/3.11.0
module load cuda/11.8.0
module load anaconda/anaconda3-2022.10

source activate py310t2cu118

# 运行TimeMixer

seq_len=96
if [ ! -d "./log_05041159_40" ]; then
    mkdir ./log_05041159_40
fi

if [ ! -d "./log_05041159_40/ETTm1" ]; then
    mkdir ./log_05041159_40/ETTm1
fi
if [ ! -d "./log_05041159_40/ETTh1" ]; then
    mkdir ./log_05041159_40/ETTh1
fi
if [ ! -d "./log_05041159_40/ETTm2" ]; then
    mkdir ./log_05041159_40/ETTm2
fi

if [ ! -d "./log_05041159_40/ETTh2" ]; then
    mkdir ./log_05041159_40/ETTh2
fi
if [ ! -d "./log_05041159_40/electricity" ]; then
    mkdir ./log_05041159_40/electricity
fi

# ettm2
#--root_path=ETT-small
#--model=micn
#--mode=regre
#--data=ETTm2
#--features=M
#--d_layers=1
#--seq_len=96
#--pred_len=96
#--d_model=128
#--d_ff=128
#--x_enc_len=7
#--x_mark_len=5
#--e_layer=2
#--itr=1
#--down_sampling_layers=3
#--down_sampling_method=avg
#--down_sampling_window=2
#--channel_independence=1
#--use_space_merge=1
#--use_fourier=1

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

#  m2 2分解核
# m2 d_model= d_ff = 128
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --root_path ETT-small \
        --model TimeMixer \
        --mode regre \
        --data ETTm2 \
        --features M \
        --d_layers 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 128 \
        --d_ff 128 \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_05041159_40/ETTm2/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done




# m1 d_model= d_ff = 128

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model TimeMixer \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_05041159_40/ETTm1/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done




#  h1 2分解核  dmodel=dff=256

for pred_len in 96  192 336 720; do
 python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
      --model TimeMixer \
      --root_path ETT-small \
      --mode regre \
      --data ETTh1 \
      --features M \
      --e_layers 2 \
      --d_layers 1 \
      --d_model 256 \
      --d_ff 256 \
      --itr 1 \
      --seq_len 96 \
      --pred_len $pred_len \
      --down_sampling_layers 3 \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --channel_independence 1 \
  > log_05041159_40/ETTh1/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done




#  h2 2分解核  d_model= d_ff = 128

for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model TimeMixer \
        --root_path ETT-small \
        --mode regre \
        --data ETTh2 \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_05041159_40/ETTh2/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done





# electricity d_model=d_ff=512

 for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model TimeMixer \
        --root_path electricity \
        --mode regre \
        --data ECL \
        --features M \
        --e_layers 3 \
        --freq h \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --batch_size 16 \
        --itr 1 \
        --learning_rate 0.0005 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
    > log_05041159_40/electricity/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
