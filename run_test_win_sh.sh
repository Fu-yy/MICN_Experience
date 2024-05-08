#for pred_len in 96 192 336 720 ; do
#   python -u  run.py \
#        --model micn \
#        --root_path electricity \
#        --mode regre \
#        --data ECL \
#        --features M \
#        --e_layers 3 \
#        --freq h \
#        --d_layers 1 \
#        --d_model 512 \
#        --d_ff 512 \
#        --seq_len 96 \
#        --pred_len $pred_len \
#        --batch_size 16 \
#        --itr 1 \
#        --learning_rate 0.0005 \
#        --down_sampling_layers 3 \
#        --down_sampling_method avg \
#        --down_sampling_window 2 \
#        --x_mark_len $electricity_x_mark_len \
#        --use_space_merge $use_space_merge \
#        --use_fourier $use_fourier \
#    > log_05072140_40/electricity/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
#  done



for pred_len in 96 192 336 720 ; do
  python -u  test_sh.py \
  > log_05072140_40/electricity/'0'0.01.log 2>&1
done