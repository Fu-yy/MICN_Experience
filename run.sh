#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH -w aiwkr1


#module load cuda/11.7.0
#module load singularity/3.11.0
module load cuda/11.8.0
module load anaconda/anaconda3-2022.10

source activate py310t2cu118


seq_len=96
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/Experence" ]; then
    mkdir ./logs/Experence
fi

if [ ! -d "./logs/ETTm1" ]; then
    mkdir ./logs/ETTm1
fi
if [ ! -d "./logs/ETTh1" ]; then
    mkdir ./logs/ETTh1
fi
if [ ! -d "./logs/ETTm2" ]; then
    mkdir ./logs/ETTm2
fi

if [ ! -d "./logs/ETTh2" ]; then
    mkdir ./logs/ETTh2
fi

if [ ! -d "./logs/electricity" ]; then
    mkdir ./logs/electricity
fi

if [ ! -d "./logs/Exchange" ]; then
    mkdir ./logs/Exchange
fi

if [ ! -d "./logs/illness" ]; then
    mkdir ./logs/illness
fi

if [ ! -d "./logs/weather" ]; then
    mkdir ./logs/weather
fi

if [ ! -d "./logs/Traffic" ]; then
    mkdir ./logs/Traffic
fi

if [ ! -d "./logs/PEMS03" ]; then
    mkdir ./logs/PEMS03
fi
if [ ! -d "./logs/PEMS04" ]; then
    mkdir ./logs/PEMS04
fi
if [ ! -d "./logs/PEMS07" ]; then
    mkdir ./logs/PEMS07
fi
if [ ! -d "./logs/PEMS08" ]; then
    mkdir ./logs/PEMS08
fi


#singularity exec --nv /mnt/nfs/data/home/1120231455/home/fuy/fuypycharm1_1.sif  nvidia-smi;\

date_ETTh1=ETTh1
date_ETTh2=ETTh2
date_ETTm1=ETTm1
date_ETTm2=ETTm2
date_exchange_rate=exchange_rate
date_national_illness=national_illness
date_weather=weather


if false; then
for pred_len in 12 24 48 96; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
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
    > logs/PEMS03/'0'_$model_name'_'PEMS03'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

for pred_len in 12 24 48 96; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
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
    > logs/PEMS04/'0'_$model_name'_'PEMS04'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

for pred_len in 12 24 48 96; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
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
    > logs/PEMS07/'0'_$model_name'_'PEMS07'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
for pred_len in 12 24 48 96; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
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
    > logs/PEMS08/'0'_$model_name'_'PEMS08'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

fi



  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path electricity \
        --mode regre \
        --data ECL \
        --features M \
        --freq h \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
    > logs/electricity/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done


  for pred_len in 24 36 48 60 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
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
    > logs/illness/'0'_$model_name'_'illness'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path weather \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
    > logs/weather/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done


  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path exchange_rate \
        --mode regre \
        --data Exchange \
        --features M \
        --freq d \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
    > logs/Exchange/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path traffic \
        --mode regre \
        --data Traffic \
        --features M \
        --freq h \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
    > logs/Traffic/'0'_$model_name'_'Traffic'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done








  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
    > logs/ETTm2/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done








  for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $pred_len \
        --use_fourier_reshape_front 0 \
    > logs/ETTm1/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTh1 \
        --features M \
        --freq t \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 256 \
        --seq_len 96 \
        --e_layers 2 \
        --pred_len $pred_len \
        --use_fourier_reshape_front 0 \
    > logs/ETTh1/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
  for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/MICN_Experience/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTh2 \
        --features M \
        --freq t \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 256 \
        --seq_len 96 \
        --label_len 96 \
        --use_fourier_reshape_front 0 \
        --pred_len $pred_len \
    > logs/ETTh2/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done


#
#  for pred_len in 96 192 336 720; do
#   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/CrossGNN/run_longExp.py \
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
#    > logs/LongForecasting/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
#
#  for pred_len in 96 192 336 720; do
#   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/CrossGNN/run_longExp.py \
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
#    > logs/LongForecasting/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
#
#  for pred_len in 96 ; do
#   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/CrossGNN/run_longExp.py \
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
#    > logs/LongForecasting/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
#
#
#
#
#  for pred_len in 96 192 336 720; do
#   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/CrossGNN/run_longExp.py \
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
#    > logs/LongForecasting/'0'_$model_name'_'exchange_rate'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
#
#
#
#
#  for pred_len in 96 192 336 720; do
#   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/CrossGNN/run_longExp.py \
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
#    > logs/LongForecasting/'0'_$model_name'_'national_illness'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
#
#
#
#
#
#  for pred_len in 96 192 336 720; do
#   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/CrossGNN/run_longExp.py \
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
#    > logs/LongForecasting/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done
