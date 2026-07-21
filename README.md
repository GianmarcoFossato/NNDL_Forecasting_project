# HyPT: Improving TimesNet for Long-Term Electricity Forecasting

This repository contains my final project for the **Neural Networks and Deep Learning** course. It is an individual project whose goal was to improve the [TimesNet](https://openreview.net/forum?id=ju_Uqw384Oq) model on **long-term time series forecasting** for the **Electricity (ECL)** dataset.

The project introduces **HyPT**, my proposed model, and benchmarks it against a set of baseline models. It is built on top of [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) by THUML — all baseline implementations come from their codebase and match the corresponding original papers. On top of TSLib, this repo adds:

- **HyPT**, my own model (`models/HyPT.py`), aimed at improving on TimesNet for this task.
- **Hyperparameter tuning** via [Optuna](https://optuna.org/) (`tune.py`, `scripts/long_term_forecast/Tuning/`).
- A completely different way to **interact with the project**: everything now runs through a hosted [molab](https://molab.marimo.io/) notebook instead of a local/Docker setup.
- A trimmed-down scope: only what's needed for **long-term forecasting on Electricity** is kept. The general-purpose TimesNet tutorial and unrelated task scaffolding have been removed.

## Run it

You don't need to install anything locally. The whole project runs through a [marimo](https://marimo.io/) notebook hosted on **molab**, using free server-side compute (a single NVIDIA RTX PRO 6000 Blackwell + 4-core CPU).


[![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_AzHXbosNtEGTcnaF8nLsHZ)


From the notebook you can:
- Fine-tune the hyperparameters of **HyPT**.
- Run one or more of the implemented models (HyPT or any baseline) on the Electricity dataset.
- View a live table comparing model performance.
- Download a `test_results.zip` for any model(s) you've trained.

> ⚠️ Model **weights are not saved/downloadable** due to their size and marimo's traffic limits.

### Data

Molab automatically downloads the official Electricity (ECL) compressed dataset records, available on [\[Google Drive\]](https://drive.google.com/drive/folders/1vE0ONyqPlym2JaaAoEe0XNDR8FS_d322?usp=drive_link), into the expected `./dataset/electricity/electricity.csv` local path framework.


## Project Structure

```
├── models/
│   ├── HyPT.py                   # My model
│   ├── TimesNet.py               # Baseline being improved upon
│   └── ...                       # Other baselines (DLinear, TimeMixer, TimeXer, iTransformer)
├── layers/
│   ├── HyPT_EncDec.py            # Layers specific to HyPT
│   └── ...                       # Shared blocks used by baselines or developed for HyPT
├── exp/
│   ├── exp_basic.py               # Experiment base class, registers models, builds flows
│   └── exp_long_term_forecasting.py  # Long-term forecasting logic
├── data_provider/
│   ├── data_factory.py            # Chooses the proper DataLoader
│   └── data_loader.py             # Electricity data reader with sliding-window logic
├── scripts/long_term_forecast/
│   ├── ECL_script/                # Train/eval scripts per model, on Electricity
│   ├── Tuning/                    # Optuna configs and scripts for HyPT
│   └── AugmentSample/             # Augmentation examples
├── utils/                         # Metrics, EarlyStopping, augmentation, masking, etc.
├── paper_test_results/            # Reference results for comparison
├── test_results/                  # Your own run outputs (downloadable as a zip via molab)
├── tune.py                        # Optuna tuning entry point
├── run.py                         # Unified entry point: parses args, dispatches tasks
├── dataset/electricity/           # Electricity dataset
└── paper                          # Folder for the paper. Includes some image resources 
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