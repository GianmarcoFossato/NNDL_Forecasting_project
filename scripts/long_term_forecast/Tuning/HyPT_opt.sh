#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

model_name="HyPT"

# Architectural Hyperparameters (Fallback defaults overridden by JSON env vars)
D_MODEL="${D_MODEL:-256}"
D_PERIOD="${D_PERIOD:-32}"          # Decoupled Branch A dimension
D_FF_MULT="${D_FF_MULT:-2}"          # d_ff multiplier (d_ff_mult in json)
D_FF=$((D_MODEL * D_FF_MULT))
N_HEADS="${N_HEADS:-8}"
E_LAYERS="${E_LAYERS:-2}"
TOP_K="${TOP_K:-3}"              # Top K periods
PATCH_LEN="${PATCH_LEN:-16}"         # Patch size for temporal embedding

# Regularization & Training (Fallback defaults overridden by JSON env vars)
DROPOUT="${DROPOUT:-0.1}"          # FFN, Patch, and Head dropout
BRANCH_DROPOUT="${BRANCH_DROPOUT:-0.15}"  # Hybrid path dropout
BRANCH_WARMUP_EPOCHS="${BRANCH_WARMUP_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"      # train_epochs in json
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-0}"
LR_ADJ="${LR_ADJ:-type3}"       # Decay schedule

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
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --lradj $LR_ADJ \
  --des 'Exp' \
  --itr 1 \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS \
  --results_subfolder "tuning" \
  --no_compile"

# --- Horizon Runs ---

python -u run.py --model_id ECL_96_96 --pred_len 96 $COMMON_ARGS
python -u run.py --model_id ECL_96_192 --pred_len 192 $COMMON_ARGS
python -u run.py --model_id ECL_96_336 --pred_len 336 $COMMON_ARGS
python -u run.py --model_id ECL_96_720 --pred_len 720 $COMMON_ARGS

echo "All ${model_name} training jobs completed."