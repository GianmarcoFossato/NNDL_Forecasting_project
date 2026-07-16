model_name=DLinear
WORKERS=0
TOTAL_TRIALS=15

# Clean old database if you want a completely fresh study sweep
rm -f optuna_study.db

# LOOP 1: Tuning configuration for Horizon 96
ID_1="ECL_96_96"
python -u run_optuna.py \
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
  --train_epochs 10 \
  --num_workers $WORKERS

# LOOP 2: Tuning configuration for Horizon 192
ID_2="ECL_96_192"
python -u run_optuna.py \
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
  --train_epochs 10 \
  --num_workers $WORKERS

# Launches the dashboard web app using the generated sqlite tracking file
optuna-dashboard sqlite:///optuna_study.db --port 8080