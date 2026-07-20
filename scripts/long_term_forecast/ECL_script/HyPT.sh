export CUDA_VISIBLE_DEVICES=0

model_name="HyPT"
BATCH_SIZE=32
WORKERS=0
BRANCH_DROPOUT=0.1
BRANCH_WARMUP_EPOCHS=3

ID_1="ECL_96_96"

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
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --branch_dropout $BRANCH_DROPOUT \
  --branch_warmup_epochs $BRANCH_WARMUP_EPOCHS \
  --des 'Exp' \
  --itr 1 \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS \
  --no_compile

ID_2="ECL_96_192"

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
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --branch_dropout $BRANCH_DROPOUT \
  --branch_warmup_epochs $BRANCH_WARMUP_EPOCHS \
  --des 'Exp' \
  --itr 1 \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS \
  --no_compile

ID_3="ECL_96_336"

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
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --branch_dropout $BRANCH_DROPOUT \
  --branch_warmup_epochs $BRANCH_WARMUP_EPOCHS \
  --des 'Exp' \
  --itr 1 \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS \
  --no_compile

ID_4="ECL_96_720"

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
  --d_model 256 \
  --d_ff 512 \
  --top_k 3 \
  --branch_dropout $BRANCH_DROPOUT \
  --branch_warmup_epochs $BRANCH_WARMUP_EPOCHS \
  --des 'Exp' \
  --itr 1 \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS \
  --no_compile

echo "All ${model_name} training jobs completed."