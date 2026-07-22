#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

model_name="HyPT"

# Architectural Hyperparameters
D_MODEL=256
D_PERIOD=16
D_FF=512
N_HEADS=8
E_LAYERS=2
TOP_K=5
PATCH_LEN=16

# Regularization & Training
DROPOUT=0.1
LEARNING_RATE=0.0005
EPOCHS=20
PATIENCE=5
BATCH_SIZE=32
WORKERS=0
LR_ADJ="type3"

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
  --learning_rate $LEARNING_RATE \
  --train_epochs $EPOCHS \
  --patience $PATIENCE \
  --lradj $LR_ADJ \
  --batch_size $BATCH_SIZE \
  --num_workers $WORKERS \
  --no_compile"

SEED=2021

for PRED_LEN in 96 192 336 720; do

  # Full Hybrid (Standard Branch Dropout = 0.1)
  python -u run.py --model_id ECL_96_${PRED_LEN}_both_s${SEED} --pred_len $PRED_LEN --ablation_mode both --branch_dropout 0.1 --seed $SEED $COMMON_ARGS

  # Full Hybrid (No Branch Dropout = 0.0)
  python -u run.py --model_id ECL_96_${PRED_LEN}_both_nodrop_s${SEED} --pred_len $PRED_LEN --ablation_mode both --branch_dropout 0.0 --seed $SEED $COMMON_ARGS

  # Baseline Floor Control (No branches)
  python -u run.py --model_id ECL_96_${PRED_LEN}_none_s${SEED} --pred_len $PRED_LEN --ablation_mode none --seed $SEED $COMMON_ARGS

  # Single Branch Baselines
  python -u run.py --model_id ECL_96_${PRED_LEN}_branch_a_s${SEED} --pred_len $PRED_LEN --ablation_mode branch_a --seed $SEED $COMMON_ARGS
  python -u run.py --model_id ECL_96_${PRED_LEN}_branch_b_s${SEED} --pred_len $PRED_LEN --ablation_mode branch_b --seed $SEED $COMMON_ARGS

done


echo "All sweep iterations of ${model_name} finished for seed ${SEED}."