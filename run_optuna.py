import optuna
import torch
import numpy as np
import os
from run import get_parser
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def objective(trial, base_args):
    """
        The Optuna objective function. It modifies base_args with trial suggestions,
        runs the training loop, and returns the evaluation metric.
        """
    # Pass the trial object dynamically into args so the training loop can use it for pruning
    base_args.trial = trial

    # General Framework Hyperparameters
    base_args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    base_args.batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Specific Model Hyperparameters
    if base_args.model == 'DLinear':
        base_args.d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    elif base_args.model == 'TimesNet':
        base_args.d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        base_args.top_k = trial.suggest_int('top_k', 1, 5)

    # Create a unique setting string
    setting = 'tune_{}_{}_{}_ft{}_sl{}_pl{}_trial{}'.format(
        base_args.model, base_args.data, base_args.model_id,
        base_args.features, base_args.seq_len, base_args.pred_len, trial.number
    )

    try:
        exp = Exp_Long_Term_Forecast(base_args)

        # Train the model (Early stopping and Optuna pruning happen inside here)
        exp.train(setting)

        # If the trial successfully finishes without being pruned, evaluate it
        exp.test(setting)

        # Extract the saved MSE
        metrics_path = os.path.join('./results/', setting, 'metrics.npy')
        if os.path.exists(metrics_path):
            # Change index to 0 if you want Optuna to optimize for MAE.
            # Keep it at 1 if you want Optuna to optimize for MSE.
            return np.load(metrics_path)[1]

        return float('inf')

    except optuna.TrialPruned:
        # Propagate exception up so Optuna flags the trial status as 'PRUNED' in the database
        raise
    except Exception as e:
        print(f"Trial failed due to error: {e}")
        return float('inf')
    finally:
        # Prevent VRAM OOM errors during sequential trials
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Device allocation
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')

    # Enforce database storage configuration to support the live Marimo dashboard
    db_file = "optuna_study.db"
    storage_url = f"sqlite:///{db_file}"
    study_name = f"tune_{args.model}_{args.model_id}_{args.pred_len}"

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)  # Starts pruning after epoch 2
    )

    # Run optimization
    study.optimize(lambda trial: objective(trial, args), n_trials=15)


if __name__ == '__main__':
    main()