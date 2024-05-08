#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH -w aiwkr3


#module load cuda/11.7.0
#module load singularity/3.11.0
module load cuda/11.8.0
module load anaconda/anaconda3-2022.10

source activate py310t2cu118

# 对out_cross_layer_test进行第二维度全连接  取消第三维度全连接  2024.4.4 14:52
# 添加第三维度全连接 同时将dff设置成2048                       2024.4.4 15：10
seq_len=96
if [ ! -d "./log_04041510_40/ETTm2" ]; then
    mkdir ./log_04041510_40/ETTm2
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
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --e_layers 2 \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 2048 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 5 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --learning_rate 0.0005 \
    > log_04041510_40/ETTm2/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done






if false; then
for pred_len in 12 24 48 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path pems \
        --mode regre \
        --data PEMS03 \
        --features M \
        --freq h \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
    > log_04041510_40/PEMS03/'0'_$model_name'_'PEMS03'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

for pred_len in 12 24 48 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path pems \
        --mode regre \
        --data PEMS04 \
        --features M \
        --freq h \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
    > log_04041510_40/PEMS04/'0'_$model_name'_'PEMS04'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

for pred_len in 12 24 48 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path pems \
        --mode regre \
        --data PEMS07 \
        --features M \
        --freq h \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
    > log_04041510_40/PEMS07/'0'_$model_name'_'PEMS07'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
for pred_len in 12 24 48 96; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path pems \
        --mode regre \
        --data PEMS08 \
        --features M \
        --freq h \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
    > log_04041510_40/PEMS08/'0'_$model_name'_'PEMS08'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done




# electricity 3分解核
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path electricity \
        --mode regre \
        --data ECL \
        --features M \
        --e_layers 3 \
        --freq h \
        --conv_kernel 12 16 \
        --decomp_kernel 17 49 127 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 512 \
        --seq_len 96 \
        --batch_size 16 \
        --learning_rate 0.0005 \
        --pred_len $pred_len \
        --itr 1 \
    > log_04041510_40/electricity/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done


  for pred_len in 24 36 48 60 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path illness \
        --mode regre \
        --data ILI \
        --features M \
        --freq d \
        --conv_kernel 18 12 \
        --d_layers 1 \
        --d_model 64 \
        --seq_len 36 \
        --label_len 36 \
        --pred_len $pred_len \
    > log_04041510_40/illness/'0'_$model_name'_'illness'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done



#  weather 3分解核
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path weather \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --conv_kernel 12 16 \
        --decomp_kernel 17 49 127 \
        --d_layers 1 \
        --e_layers 3 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
    > log_04041510_40/weather/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done




#  traffic 4分解核
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path traffic \
        --mode regre \
        --data Traffic \
        --features M \
        --freq h \
        --e_layers 4 \
        --conv_kernel 12 16 \
        --decomp_kernel 17 49 127 511 \
        --d_layers 1 \
        --d_model 512 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --d_ff 512 \
        --seq_len 96 \
        --itr 1 \
        --pred_len $pred_len \
    > log_04041510_40/Traffic/'0'_$model_name'_'Traffic'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done



#  Exchange 2分解核

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path exchange_rate \
        --mode regre \
        --data Exchange \
        --features M \
        --freq d \
        --conv_kernel 12 16 \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --pred_len $pred_len \
    > log_04041510_40/Exchange/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done




fi


if false; then
  #  m1 2分解核
  for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --e_layers 2 \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
    > log_04041510_40/ETTm1/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  #  h1 2分解核
  for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTh1 \
        --features M \
        --freq t \
        --e_layers 2 \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 256 \
        --d_ff 256 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 4 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04041510_40/ETTh1/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  #  h2 2分解核
  for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTh2 \
        --features M \
        --freq t \
        --e_layers 2 \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len 4 \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
    > log_04041510_40/ETTh2/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
fi

#
#  for pred_len in 96 192 336 720; do
#   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/CrossGNN/run_longExp.py \
#    --train_epochs 10 \
#    --is_training 1 \
#    --e_layers 1 \
#    --label_len 96 \
#    --gpu 0 \
#    --root_path ./dataset/ETT-small/ \
#    --data_path ETTh2.csv \
#    --model_id ETTh2'_'$seq_len'_'$pred_len \
#    --model FGN \
#    --data ETTh2 \
#    --features M \
#    --seq_len $seq_len \
#    --pred_len $pred_len \
#    --enc_in 7 \
#    --des 'Exp' \
#    --itr 1 \
#    --top_k 5 \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --d_ff 32 \
#    --pre_length 12 \
#    --embed_size 128 \
#    --feature_size 140 \
#    --seq_length 12 \
#    --hidden_size 256 \
#    --hard_thresholding_fraction 1 \
#    --hidden_size_factor 1 \
#    --sparsity_threshold 0.01 \
#    --d_model 7 \
#    > log_04041510_40/LongForecasting/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
#
#  for pred_len in 96 192 336 720; do
#   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/CrossGNN/run_longExp.py \
#    --train_epochs 10 \
#    --is_training 1 \
#    --e_layers 1 \
#    --label_len 96 \
#    --gpu 0 \
#    --root_path ./dataset/ETT-small/ \
#    --data_path ETTm1.csv \
#    --model_id ETTm1'_'$seq_len'_'$pred_len \
#    --model FGN \
#    --data ETTm1 \
#    --features M \
#    --seq_len $seq_len \
#    --pred_len $pred_len \
#    --enc_in 7 \
#    --des 'Exp' \
#    --itr 1 \
#    --top_k 5 \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --d_ff 32 \
#    --pre_length 12 \
#    --embed_size 128 \
#    --feature_size 140 \
#    --seq_length 12 \
#    --hidden_size 256 \
#    --hard_thresholding_fraction 1 \
#    --hidden_size_factor 1 \
#    --sparsity_threshold 0.01 \
#    --d_model 7 \
#    > log_04041510_40/LongForecasting/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
#
#  for pred_len in 96 ; do
#   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/CrossGNN/run_longExp.py \
#    --train_epochs 10 \
#    --is_training 1 \
#    --e_layers 1 \
#    --label_len 96 \
#    --gpu 0 \
#    --root_path ./dataset/ETT-small/ \
#    --data_path ETTm2.csv \
#    --model_id ETTm2'_'$seq_len'_'$pred_len \
#    --model FGN \
#    --data ETTm2 \
#    --features M \
#    --seq_len $seq_len \
#    --pred_len $pred_len \
#    --enc_in 7 \
#    --des 'Exp' \
#    --itr 1 \
#    --top_k 5 \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --d_ff 32 \
#    --pre_length 12 \
#    --embed_size 128 \
#    --feature_size 140 \
#    --seq_length 12 \
#    --hidden_size 256 \
#    --hard_thresholding_fraction 1 \
#    --hidden_size_factor 1 \
#    --sparsity_threshold 0.01 \
#    --d_model 7 \
#    > log_04041510_40/LongForecasting/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
#
#
#
#
#  for pred_len in 96 192 336 720; do
#   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/CrossGNN/run_longExp.py \
#    --train_epochs 10 \
#    --is_training 1 \
#    --e_layers 1 \
#    --label_len 96 \
#    --gpu 0 \
#    --root_path ./dataset/exchange_rate/ \
#    --data_path exchange_rate.csv \
#    --model_id exchange_rate'_'$seq_len'_'$pred_len \
#    --model FGN \
#    --data exchange_rate \
#    --features M \
#    --seq_len $seq_len \
#    --pred_len $pred_len \
#    --enc_in 7 \
#    --des 'Exp' \
#    --itr 1 \
#    --top_k 5 \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --d_ff 32 \
#    --pre_length 12 \
#    --embed_size 128 \
#    --feature_size 140 \
#    --seq_length 12 \
#    --hidden_size 256 \
#    --hard_thresholding_fraction 1 \
#    --hidden_size_factor 1 \
#    --sparsity_threshold 0.01 \
#    --d_model 7 \
#    > log_04041510_40/LongForecasting/'0'_$model_name'_'exchange_rate'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
#
#
#
#
#  for pred_len in 96 192 336 720; do
#   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/CrossGNN/run_longExp.py \
#    --train_epochs 10 \
#    --is_training 1 \
#    --e_layers 1 \
#    --label_len 96 \
#    --gpu 0 \
#    --root_path ./dataset/ilness/ \
#    --data_path national_illness.csv \
#    --model_id national_illness'_'$seq_len'_'$pred_len \
#    --model FGN \
#    --data national_illness \
#    --features M \
#    --seq_len $seq_len \
#    --pred_len $pred_len \
#    --enc_in 7 \
#    --des 'Exp' \
#    --itr 1 \
#    --top_k 5 \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --d_ff 32 \
#    --pre_length 12 \
#    --embed_size 128 \
#    --feature_size 140 \
#    --seq_length 12 \
#    --hidden_size 256 \
#    --hard_thresholding_fraction 1 \
#    --hidden_size_factor 1 \
#    --sparsity_threshold 0.01 \
#    --d_model 7 \
#    > log_04041510_40/LongForecasting/'0'_$model_name'_'national_illness'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
#
#
#
#
#
#  for pred_len in 96 192 336 720; do
#   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/CrossGNN/run_longExp.py \
#    --train_epochs 10 \
#    --is_training 1 \
#    --e_layers 1 \
#    --label_len 96 \
#    --gpu 0 \
#    --root_path ./dataset/weather/ \
#    --data_path weather01.csv \
#    --model_id weather'_'$seq_len'_'$pred_len \
#    --model FGN \
#    --data weather \
#    --features M \
#    --seq_len $seq_len \
#    --pred_len $pred_len \
#    --enc_in 7 \
#    --des 'Exp' \
#    --itr 1 \
#    --top_k 5 \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --d_ff 32 \
#    --pre_length 12 \
#    --embed_size 128 \
#    --feature_size 140 \
#    --seq_length 12 \
#    --hidden_size 256 \
#    --hard_thresholding_fraction 1 \
#    --hidden_size_factor 1 \
#    --sparsity_threshold 0.01 \
#    --d_model 7 \
#    > log_04041510_40/LongForecasting/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
