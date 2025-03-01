export CUDA_VISIBLE_DEVICES=0

model_name=MGDformer

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --model_id weather_96_96 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 1 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --train_epochs 20 \
#  --learning_rate 0.0005 \
#  --itr 1


#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --model_id weather_96_192 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512\
#  --learning_rate 0.0005 \
#  --itr 1
#
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --model_id weather_96_336 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 1 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512\
#  --learning_rate 0.0005 \
#  --itr 1
#
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --model_id weather_96_720 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512\
#  --learning_rate 0.0005 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_96 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_192 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 2 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_336 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 2 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1
##
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_720 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 2 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --learning_rate 0.0001 \
#  --itr 1

#  python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_96 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_192 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 2 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1
##
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_336 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 2 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_720 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 2 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_96 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_192 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_336 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_720 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --learning_rate 0.0001 \
#  --itr 1



#  python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_96 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_192 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_336 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_720 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --learning_rate 0.0001 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/electricity/ \
#  --data_path electricity.csv \
#  --model_id ECL_96_96 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 3 \
#  --enc_in 321 \
#  --dec_in 321 \
#  --c_out 321 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --batch_size 16 \
#  --learning_rate 0.001 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/electricity/ \
#  --data_path electricity.csv \
#  --model_id ECL_96_192 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 3 \
#  --enc_in 321 \
#  --dec_in 321 \
#  --c_out 321 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --batch_size 16 \
#  --learning_rate 0.001 \
#  --itr 1
#
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/electricity/ \
#  --data_path electricity.csv \
#  --model_id ECL_96_336 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 3 \
#  --enc_in 321 \
#  --dec_in 321 \
#  --c_out 321 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --batch_size 16 \
#  --learning_rate 0.001 \
#  --itr 1
#
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/electricity/ \
#  --data_path electricity.csv \
#  --model_id ECL_96_720 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 3 \
#  --enc_in 321 \
#  --dec_in 321 \
#  --c_out 321 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --batch_size 16 \
#  --learning_rate 0.001 \
#  --itr 1




#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --model_id Exchange_96_96 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 1 \
#  --enc_in 8 \
#  --dec_in 8 \
#  --c_out 8 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 256 \
#  --learning_rate 0.00003 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --model_id Exchange_96_192 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --enc_in 8 \
#  --dec_in 8 \
#  --c_out 8 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 256 \
#  --learning_rate 0.00003 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --model_id Exchange_96_336 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 1 \
#  --enc_in 8 \
#  --dec_in 8 \
#  --c_out 8 \
#  --des 'Exp' \
#  --itr 1 \
#  --d_model 512 \
#  --learning_rate 0.00003 \
#  --d_ff 512 \
##  --train_epochs 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --model_id Exchange_96_720 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --enc_in 8 \
#  --dec_in 8 \
#  --c_out 8 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --learning_rate 0.00003 \
#  --itr 1



  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic_test.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0025 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic_test.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0025 \
  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic_test.csv \
#  --model_id traffic_96_336 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 4 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --d_model 512\
#  --d_ff 512 \
#  --batch_size 16 \
#  --learning_rate 0.0025 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic_test.csv \
#  --model_id traffic_96_720 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 4 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --batch_size 16 \
#  --learning_rate 0.0025 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/Solar/ \
#  --data_path solar_AL.csv \
#  --model_id solar_96_96 \
#  --model $model_name \
#  --data Solar \
#  --features M \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --enc_in 137 \
#  --dec_in 137 \
#  --c_out 137 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --learning_rate 0.0005 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/Solar/ \
#  --data_path solar_AL.csv \
#  --model_id solar_96_192 \
#  --model $model_name \
#  --data Solar \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 2 \
#  --enc_in 137 \
#  --dec_in 137 \
#  --c_out 137 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --learning_rate 0.0005 \
#  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/Solar/ \
#  --data_path solar_AL.csv \
#  --model_id solar_96_336 \
#  --model $model_name \
#  --data Solar \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 2 \
#  --enc_in 137 \
#  --dec_in 137 \
#  --c_out 137 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --learning_rate 0.0005 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/Solar/ \
#  --data_path solar_AL.csv \
#  --model_id solar_96_720 \
#  --model $model_name \
#  --data Solar \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 2 \
#  --enc_in 137 \
#  --dec_in 137 \
#  --c_out 137 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --learning_rate 0.0005 \
#  --itr 1

