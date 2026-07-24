#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

model_name="HyPT"

# Architectural Hyperparameters (Optimized)
D_MODEL=256
D_PERIOD=32          # Decoupled Branch A dimension
D_FF=512             # d_model * d_ff_mult (256 * 2)
N_HEADS=8
E_LAYERS=2
TOP_K=3              # Top K periods
PATCH_LEN=16         # Patch size for temporal embedding

# Regularization & Training (Optimized)
DROPOUT=0.1          # FFN, Patch, and Head dropout
BRANCH_DROPOUT=0.15  # Hybrid path dropout
BRANCH_WARMUP_EPOCHS=2
LEARNING_RATE=0.001
EPOCHS=20
PATIENCE=5
BATCH_SIZE=16
WORKERS=0
LR_ADJ="type3"       # Gentler decay schedule for multi-channel convergence

# Common arguments string
COMMON_ARGS="--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --e_layers $E_LAYERS \
  --d_layers 1 \
  --factor 3 \
  --d_model $D_MODEL \
  --d_period $D_PERIOD \
  --d_ff $D_FF \
  --n_heads $N_HEADS \
  --patch_len $PATCH_LEN \
  --top_k $TOP_K \
  --dropout $DROPOUT \
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
  --no_compile"

# --- Horizon Runs ---

python -u run.py --model_id ECL_96_96 --pred_len 96 $COMMON_ARGS
python -u run.py --model_id ECL_96_192 --pred_len 192 $COMMON_ARGS
python -u run.py --model_id ECL_96_336 --pred_len 336 $COMMON_ARGS
python -u run.py --model_id ECL_96_720 --pred_len 720 $COMMON_ARGS

echo "All ${model_name} training jobs completed."