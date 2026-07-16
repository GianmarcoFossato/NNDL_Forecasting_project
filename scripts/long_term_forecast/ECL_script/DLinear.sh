# export CUDA_VISIBLE_DEVICES=0
model_name=DLinear

WORKERS=0

# RUN 1: ECL_96_96
ID_1="ECL_96_96"
SETTING_1="long_term_forecast_${ID_1}_${model_name}_custom_ftM_sl96_ll48_pl96_el2_dl1_fc3_Exp_0"
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
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --num_workers $WORKERS > "./test_results/${SETTING_1}/output.log" 2>&1

# RUN 2: ECL_96_192
ID_2="ECL_96_192"
SETTING_2="long_term_forecast_${ID_2}_${model_name}_custom_ftM_sl96_ll48_pl192_el2_dl1_fc3_Exp_0"
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
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS > "./test_results/${SETTING_2}/output.log" 2>&1

# RUN 3: ECL_96_336
ID_3="ECL_96_336"
SETTING_3="long_term_forecast_${ID_3}_${model_name}_custom_ftM_sl96_ll48_pl336_el2_dl1_fc3_Exp_0"
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
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --num_workers $WORKERS > "./test_results/${SETTING_3}/output.log" 2>&1

# RUN 4: ECL_96_720
ID_4="ECL_96_720"
SETTING_4="long_term_forecast_${ID_4}_${model_name}_custom_ftM_sl96_ll48_pl720_el2_dl1_fc3_Exp_0"
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
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS > "./test_results/${SETTING_4}/output.log" 2>&1

echo "All DLinear training jobs completed."