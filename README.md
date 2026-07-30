# HyPT: Improving TimesNet for Long-Term Electricity Forecasting

This repository contains my final project for the **Neural Networks and Deep Learning** course. It is an individual project whose goal was to improve the [TimesNet](https://openreview.net/forum?id=ju_Uqw384Oq) model on **long-term time series forecasting** for the **Electricity** dataset.

The project introduces **HyPT**, my proposed model, and benchmarks it against a set of baseline models. It is built on top of [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) by THUML — all baseline implementations come from their codebase and match the corresponding original papers. On top of TSLib, this repo adds:

- **HyPT**, my own model (`models/HyPT.py`), aimed at improving on TimesNet for this task.
- **Hyperparameter tuning** via [Optuna](https://optuna.org/) (`tune.py`, `scripts/long_term_forecast/Tuning/`).
- A completely different way to **interact with the project**: everything now runs through a hosted [molab](https://molab.marimo.io/) notebook instead of a local/Docker setup.
- A trimmed-down scope: only what's needed for **long-term forecasting on Electricity** is kept. The general-purpose TimesNet tutorial and unrelated task scaffolding have been removed.

## Run it

You don't need to install anything locally. The whole project runs through a [marimo](https://marimo.io/) notebook hosted on **molab**, using free server-side compute (a single NVIDIA RTX PRO 6000 Blackwell + 4-core CPU).


[![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_zvjVSJCj48n58ntjd3Zwbr)

From the notebook you can:
- Fine-tune the hyperparameters of **HyPT**.
- Run one or more of the implemented models (HyPT or any baseline) on the Electricity dataset.
- View a live table comparing model performance.
- Download a `test_results.zip` for logs, loss curves, etc. of model(s) you've trained.

> ⚠️ Model **weights are not saved/downloadable** due to their size and molab's traffic limits.

### Data

Molab automatically downloads the official Electricity (ECL) compressed dataset records, available on [\[Google Drive\]](https://drive.google.com/drive/folders/1vE0ONyqPlym2JaaAoEe0XNDR8FS_d322?usp=drive_link), into the expected `./dataset/electricity/electricity.csv` local path framework.


## Project Structure

```
├── NNDL Project.py                # Marimo notebook implementation backup for molab
├── README.md                      # Project documentation
├── requirements.txt               # Python package dependencies
├── run.py                         # Unified entry point: parses CLI args and dispatches tasks
├── tune.py                        # Optuna hyperparameter optimization entry point
│
├── data_provider/                 # Data loading and batching pipeline
│   ├── data_factory.py            # DataLoader builder for dataset selection
│   └── data_loader.py             # Dataset Electricity reader with sliding-window logic
│
├── dataset/                       # Local storage for datasets
│   └── electricity/
│       └── electricity.csv        # Electricity benchmark dataset (321 variates)
│
├── exp/                           # Experiment execution drivers
│   ├── exp_basic.py               # Base class registering models and device placement
│   └── exp_long_term_forecasting.py  # Long-term forecasting train/val/test workflow
│
├── layers/                        # Neural network layers and building blocks
│   ├── HyPT_EncDec.py             # Hybrid Encoder Layer (Branch A, Branch B, Gated Fusion)
│   ├── ConvNeXtBlock2D.py         # ConvNeXt 2D depthwise block for periodicity modeling
│   ├── Embed.py                   # PatchTST-style tokenization and positional embeddings
│   ├── RevIN.py                   # Reversible Instance Normalization layer
│   ├── SelfAttention_Family.py    # Multi-head attention implementations
│   └── ...                        # Additional baseline layers from TSLib
│
├── models/                        # Forecasting model implementations
│   ├── HyPT.py                    # Proposed Hybrid Period-Transformer (HyPT) model
│   ├── TimesNet.py                # 2D temporal variation baseline
│   ├── iTransformer.py            # Inverted cross-variate attention baseline
│   ├── TimeXer.py                 # Patch & variate attention baseline
│   ├── TimeMixer.py               # Multiscale MLP-mixing baseline
│   ├── DLinear.py                 # Linear decomposition baseline
│   └── PatchTST.py                # Channel-independent patch Transformer baseline
│
├── report/                        # Folder for all report-related files
│   ├── TeX source/                # LaTeX and images of the report 
│   └── report.pdf/                # Report
│
├── paper_test_results/            # Reference benchmark results produced for the paper
│   ├── ablation/                  # Seed-wise ablation logs (Seeds 2021, 2022, 2023)
│   ├── evaluation/                # Model evaluation logs across forecast horizons
│   └── tuning/                    # Optuna hyperparameter study databases and logs
│
├── scripts/                       # Shell scripts for experiments
│   └── long_term_forecast/
│       ├── Ablation/              # Scripts running component ablation studies
│       ├── ECL_script/            # Model training/evaluation scripts on Electricity
│       └── Tuning/                # Optuna HPO search configs and execution scripts
│
├── test_results/                  # Directory for local execution output logs and plots
└── utils/                         # Metrics, EarlyStopping, time features, and tools
```

### Architecture, in short

- **E2E flow**: configure via `scripts/long_term_forecast/ECL_script/*.sh` (or the molab notebook) → `run.py` parses arguments and dispatches to `Exp_Long_Term_Forecast` → the experiment builds the Electricity dataset via `data_provider`, instantiates the chosen model from `models/` (HyPT or a baseline), and drives train/val/test using `utils/` → metrics and outputs are written to `./test_results` and `./checkpoints`.
- **HyPT** (`models/HyPT.py`, `layers/HyPT_EncDec.py`) is the model I designed to improve on TimesNet for this task; all other models in `models/` are baselines carried over from TSLib for comparison.
- **Tuning** (`tune.py`, `scripts/long_term_forecast/Tuning/`) wraps the experiment loop in an Optuna study to search HyPT's hyperparameters.

## Acknowledgements

This project is built on top of **Time-Series-Library (TSLib)**. Baseline models and much of the experiment/data pipeline come directly from their implementation.

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