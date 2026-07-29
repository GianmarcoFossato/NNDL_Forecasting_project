export CUDA_VISIBLE_DEVICES=0

model_name="HyPT"

for aug in jitter scaling permutation magwarp timewarp windowslice windowwarp rotation spawner dtwwarp shapedtwwarp discdtw discsdtw
do
for pred_len in 96 192 336 720
do
echo using augmentation: ${aug}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id HyPT_aug_96_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --d_model 256 \
  --d_period 32 \
  --d_ff 512 \
  --n_heads 8 \
  --patch_len 16 \
  --top_k 3 \
  --dropout 0.1 \
  --branch_dropout 0.15 \
  --branch_warmup_epochs 2 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 3 \
  --batch_size 16 \
  --lradj type3 \
  --num_workers 0 \
  --des 'Exp' \
  --itr 1 \
  --augmentation_ratio 1 \
  --no_compile \
  --${aug}
done
done