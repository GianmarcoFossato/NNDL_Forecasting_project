#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

model_name="HyPT"

# Architectural Hyperparameters
D_MODEL=256
D_PERIOD=16          # Decoupled Branch A dimension (1/4 of d_model)
D_FF=512
N_HEADS=8
E_LAYERS=2
TOP_K=5

# Training Hyperparameters
BATCH_SIZE=32
WORKERS=0
BRANCH_DROPOUT=0.1
BRANCH_WARMUP_EPOCHS=3
LEARNING_RATE=0.0005
EPOCHS=20
PATIENCE=5
LR_ADJ="type3"       # Gentler decay schedule for multi-channel convergence

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
  --e_layers $E_LAYERS \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model $D_MODEL \
  --d_period $D_PERIOD \
  --d_ff $D_FF \
  --n_heads $N_HEADS \
  --top_k $TOP_K \
  --branch_dropout $BRANCH_DROPOUT \
  --branch_warmup_epochs $BRANCH_WARMUP_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --train_epochs $EPOCHS \
  --patience $PATIENCE \
  --lradj $LR_ADJ \
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
  --e_layers $E_LAYERS \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model $D_MODEL \
  --d_period $D_PERIOD \
  --d_ff $D_FF \
  --n_heads $N_HEADS \
  --top_k $TOP_K \
  --branch_dropout $BRANCH_DROPOUT \
  --branch_warmup_epochs $BRANCH_WARMUP_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --train_epochs $EPOCHS \
  --patience $PATIENCE \
  --lradj $LR_ADJ \
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
  --e_layers $E_LAYERS \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model $D_MODEL \
  --d_period $D_PERIOD \
  --d_ff $D_FF \
  --n_heads $N_HEADS \
  --top_k $TOP_K \
  --branch_dropout $BRANCH_DROPOUT \
  --branch_warmup_epochs $BRANCH_WARMUP_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --train_epochs $EPOCHS \
  --patience $PATIENCE \
  --lradj $LR_ADJ \
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
  --e_layers $E_LAYERS \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model $D_MODEL \
  --d_period $D_PERIOD \
  --d_ff $D_FF \
  --n_heads $N_HEADS \
  --top_k 3 \
  --branch_dropout $BRANCH_DROPOUT \
  --branch_warmup_epochs $BRANCH_WARMUP_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --train_epochs $EPOCHS \
  --patience $PATIENCE \
  --lradj $LR_ADJ \
  --des 'Exp' \
  --itr 1 \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS \
  --no_compile

echo "All ${model_name} training jobs completed."