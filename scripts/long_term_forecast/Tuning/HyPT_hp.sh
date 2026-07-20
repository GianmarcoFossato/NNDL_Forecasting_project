export CUDA_VISIBLE_DEVICES=0

python -u tune.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model HyPT \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --train_epochs 20 \
  --patience 5 \
  --num_workers 0 \
  --path_to_hp_config scripts/long_term_forecast/Tuning/HyPT_config.json