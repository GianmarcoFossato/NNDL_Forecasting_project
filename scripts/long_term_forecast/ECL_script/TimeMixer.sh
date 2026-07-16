# export CUDA_VISIBLE_DEVICES=0
model_name=TimeMixer

BATCH_SIZE=32
WORKERS=0

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=20
patience=10

# RUN 1: ECL_96_96
ID_1="ECL_96_96"
SETTING_1="long_term_forecast_${ID_1}_${model_name}_custom_ftM_sl96_ll0_pl96_dm16_el3_df32_Exp_0_revin0"
mkdir -p "./test_results/${SETTING_1}"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $ID_1 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS 2>&1 | tee "./test_results/${SETTING_1}/output.log"

# RUN 2: ECL_96_192
ID_2="ECL_96_192"
SETTING_2="long_term_forecast_${ID_2}_${model_name}_custom_ftM_sl96_ll0_pl192_dm16_el3_df32_Exp_0_revin0"
mkdir -p "./test_results/${SETTING_2}"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $ID_2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS 2>&1 | tee "./test_results/${SETTING_2}/output.log"

# RUN 3: ECL_96_336
ID_3="ECL_96_336"
SETTING_3="long_term_forecast_${ID_3}_${model_name}_custom_ftM_sl96_ll0_pl336_dm16_el3_df32_Exp_0_revin0"
mkdir -p "./test_results/${SETTING_3}"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $ID_3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS 2>&1 | tee "./test_results/${SETTING_3}/output.log"

# RUN 4: ECL_96_720
ID_4="ECL_96_720"
SETTING_4="long_term_forecast_${ID_4}_${model_name}_custom_ftM_sl96_ll0_pl720_dm16_el3_df32_Exp_0_revin0"
mkdir -p "./test_results/${SETTING_4}"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $ID_4 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS 2>&1 | tee "./test_results/${SETTING_4}/output.log"

echo "All TimeMixer training jobs completed."