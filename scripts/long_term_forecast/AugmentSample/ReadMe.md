# Augmentation Feature

For testing augmentation techniques in `Time-Series-Library`.

For now, we have embedded several augmentation methods
in this repo.

```
The Implemented Augmentation Methods
1. jitter 
2. scaling 
3. permutation 
4. magwarp 
5. timewarp 
6. windowslice 
7. windowwarp 
8. rotation 
9. spawner 
10. dtwwarp 
11. shapedtwwarp 
12. discdtw
```

## Usage

We test multiple augmentation algorithms on `Electricity` 
using `HyPT` model.

```shell
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
```

Where parameter `augmentation_ratio` represents how many
times do we want to perform our augmentation method.
Parameter `${aug}` represents a string of augmentation
type label. 

The example here only perform augmentation once, so we
can set `augmentation_ratio` to `1`, followed by one
augmentation type label. Trivially, you can set 
`augmentation_ratio` to an integer `num` followed by 
`num` augmentation type labels.

The augmentation code obeys the same prototype of 
`Time-Series-Library`.