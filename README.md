# Time Series Library (TSLib)

## Getting Started

### Prepare Data
You can obtain the well-preprocessed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) or [[Hugging Face]](https://huggingface.co/datasets/thuml/Time-Series-Library). Then place the downloaded data in the folder `./dataset`.

### Installation
1. Clone this repository.
   ```bash
   git clone https://github.com/thuml/Time-Series-Library.git
   cd Time-Series-Library
   ```

2. Create a new Conda environment.
   ```bash
   conda create -n tslib python=3.11
   conda activate tslib
   ```

3. Install Core Dependencies
   > ⚠️ **CUDA Compatibility Notice**
   > The torch prebuilt package is **CUDA-version specific**. (See https://pytorch.org/get-started/previous-versions/)
   > Please make sure to install the package that matches your local CUDA version (e.g., `cu118` or `cu121`).
   > Recommended: torch==2.5.1

   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

   pip install -r requirements.txt
   ```

4. Install Dependencies for Mamba Model (Required for Time-Series-Library/models/Mamba.py)
   > ⚠️ **Linux only**
   > ⚠️ **CUDA Compatibility Notice**
   > The prebuilt Mamba wheel is **CUDA-version specific**.
   > Please make sure to install the wheel that matches your local CUDA version
   > (e.g., `cu11` or `cu12`). Installing a mismatched version may result in
   > runtime errors or import failures.

   Example for **CUDA 12**:

   ```bash
   pip install https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
   ```

5. Install Dependencies for Moirai Model (Required for Time-Series-Library/models/Moirai.py)
   ```bash
   pip install uni2ts --no-deps
   ```

### Docker Deployment
```bash
# Build and start the Docker container in detached mode
docker compose -f 'Time-Series-Library/docker-compose.yml' up -d --build

# Download / place the dataset into a newly created folder ./dataset at the repository root
mkdir -p dataset  # create the dataset directory

# Copy the local dataset into the container at /workspace/dataset
docker cp ./dataset tslib:/workspace/dataset

# Enter the running container to continue training / evaluation
docker exec -it tslib bash

# Switch to the workspace directory inside the container
cd /workspace

# Run zero-shot forecasting with the pre-trained Moirai model
python -u run.py \
  --task_name zero_shot_forecast \   # task type: zero-shot forecasting
  --is_training 0 \                  # 0 = inference only (no training)
  --root_path ./dataset/ETT-small/ \ # root directory of the dataset
  --data_path ETTh1.csv \            # dataset file name
  --model_id ETTh1_512_96 \          # experiment/model identifier
  --model Moirai \                   # model name (TimesFM / Moirai)
  --data ETTh1 \                     # dataset name
  --features M \                     # multivariate forecasting
  --seq_len 512 \                    # input sequence length
  --pred_len 96 \                    # prediction horizon
  --enc_in 7 \                       # number of input variables
  --des 'Exp' \                      # experiment description
  --itr 1                             # number of runs
```


### Quick Test

Quick test for all 5 tasks (1 epoch each):

```bash
# Run quick tests for all 5 tasks
export CUDA_VISIBLE_DEVICES=0

# 1. Long-term forecasting
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_long --model DLinear --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --train_epochs 1 --num_workers 2

# 2. Short-term forecasting (using ETT dataset with shorter prediction length)
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_short --model TimesNet --data ETTh1 --features M --seq_len 24 --label_len 12 --pred_len 24 --e_layers 2 --d_layers 1 --d_model 16 --d_ff 32 --enc_in 7 --dec_in 7 --c_out 7 --top_k 5 --train_epochs 1 --num_workers 2

# 3. Imputation
python -u run.py --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_imp --model TimesNet --data ETTh1 --features M --seq_len 96 --e_layers 2 --d_layers 1 --d_model 16 --d_ff 32 --enc_in 7 --dec_in 7 --c_out 7 --top_k 3 --train_epochs 1 --num_workers 2 --label_len 0 --pred_len 0 --mask_rate 0.125 --learning_rate 0.001

# 4. Anomaly detection
python -u run.py --task_name anomaly_detection --is_training 1 --root_path ./dataset/PSM --model_id test_ad --model TimesNet --data PSM --features M --seq_len 100 --pred_len 0 --d_model 64 --d_ff 64 --e_layers 2 --enc_in 25 --c_out 25 --anomaly_ratio 1.0 --top_k 3 --train_epochs 1 --batch_size 128 --num_workers 2

# 5. Classification
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/Heartbeat/ --model_id Heartbeat --model TimesNet --data UEA --e_layers 2 --d_layers 1 --factor 3 --d_model 64 --d_ff 128 --top_k 3 --train_epochs 1 --batch_size 16 --learning_rate 0.001 --num_workers 0
```


### Train and Evaluate

We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

> ⚠️ Some scripts have `CUDA_VISIBLE_DEVICES` set by default. Please modify or remove this setting according to your actual GPU configuration, otherwise it may prevent GPU usage.

```bash
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```

### Develop Your Own Model
- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Create the corresponding scripts under the folder `./scripts`.


### Inspect the project structure:

```
Time-Series-Library/
├── README.md                     # Official README with tasks, leaderboard, usage
├── requirements.txt              # pip dependency list for quick environment setup
├── LICENSE / CONTRIBUTING.md     # Upstream license and contribution guide
├── run.py                        # Unified entry that parses args and dispatches tasks
├── exp/                          # Task pipelines wrapping train/val/test
│   ├── exp_basic.py              # Experiment base class, registers models, builds flows
│   └── exp_long_term_forecasting.py    # Long-term forecasting logic
├── data_provider/                # Dataset loaders and splits
│   ├── data_factory.py           # Chooses the proper DataLoader per task
│   ├── data_loader.py            # Generic TS reader with sliding-window logic
│   └── __init__.py               # Exposes factory interfaces upward
├── models/                       # All model implementations
│   ├── TimesNet.py, TimeMixer.py # Main forecasting models
│   └── __init__.py               # Enables name-based instantiation inside exp
├── layers/                       # Reusable attention / conv / embedding blocks
│   ├── Transformer_EncDec.py     # Transformer stacks
│   ├── AutoCorrelation.py        # Auto-correlation operator
│   ├── MultiWaveletCorrelation.py# Frequency-domain unit
│   └── Embed.py etc.             # Shared primitives
├── utils/                        # Utility toolbox
│   ├── metrics.py                # MSE / MAE / DTW and other metrics
│   ├── tools.py                  # General helpers such as EarlyStopping
│   ├── augmentation.py           # Augmentations for classification / detection
│   ├── print_args.py             # Unified argument printer
│   └── masking.py / losses.py    # Task-specific helpers
└── scripts/                      # Bash recipes for reproducible experiments
    └── long_term_forecast/       # Long-term forecasting per dataset/model
```

### Understand the project architecture:

- **E2E flow**: configure experiments via `scripts/*.sh` → run `python run.py ...` → `run.py` parses arguments and selects the proper `Exp_*` via `task_name` → the experiment builds datasets through `data_provider`, instantiates networks from `models`, and drives train/val/test with utilities in `utils` → metrics and checkpoints are written to `./checkpoints`.
- **Experiment layer (`exp/`)**: `Exp_Basic` registers models and devices; subclasses implement `_get_data`, `train`, and `test` to encapsulate task-specific differences so the same model can be reused.
- **Model & layer layer (`models/` + `layers/`)**: model files define architectures, while reusable attention/conv/frequency components live in `layers/` to minimize duplication.
- **Data layer (`data_provider/`)**: `data_factory` returns the correct `Dataset/DataLoader`; `data_loader` handles windowing, masking, and sampling, with arguments controlling window length, missing ratio, anomaly ratio, etc.
- **Script layer (`scripts/`)**: bash scripts capture paper configurations (dataset, window, model, GPU) for reproducibility and serve as templates for custom runs.
- **Utility layer (`utils/`)**: `metrics` centralizes evaluation, `tools` bundles essentials like `EarlyStopping` and `adjust_learning_rate`, while `augmentation`/`masking` cover task-specific preprocessing.
- **Learning path**: recommended reading order is `scripts -> run.py -> exp/exp_basic.py -> corresponding Exp subclass -> data_provider -> models`, using `tutorial/TimesNet_tutorial.ipynb` as a guided walkthrough before diving deeper.

## Citation

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}

@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```