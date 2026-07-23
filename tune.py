import os
import torch
import optuna
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
from run import init_parser, OutputLogger
from optuna.pruners import HyperbandPruner
from functools import partial
import json
import shutil
import copy


def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_trials_callback(study, trial, args):
    subfolder = getattr(args, 'results_subfolder', 'evaluation')
    model_dir = os.path.join('test_results', subfolder, f'hp_search_{args.model_id}_{args.model}')
    os.makedirs(model_dir, exist_ok=True)

    df = study.trials_dataframe()
    last_trial_file = os.path.join(model_dir, f'trials_{args.model}_{args.model_id}.csv')
    df.to_csv(last_trial_file, index=False)


def suggest_params(trial, args, hp_configs):
    """
    Use Optuna to suggest hyperparameters based on the config file.
    Ensures structural dependencies (e.g., d_ff >= d_model) are respected.
    """
    parameters = hp_configs['parameters'].copy()

    # Override parameters for special model cases
    special_cases = hp_configs.get('special_cases', {})
    if args.model in special_cases:
        special_parameters = special_cases[args.model].get('parameters', {})
        for param, special_values in special_parameters.items():
            parameters[param] = special_values

    sampled = {}

    # Sample d_model first
    if 'd_model' in parameters:
        p_kwargs = parameters['d_model']['kwargs']
        p_type = parameters['d_model']['type']
        if p_type == 'categorical':
            d_model_val = trial.suggest_categorical('d_model', **p_kwargs)
        elif p_type == 'int':
            d_model_val = trial.suggest_int('d_model', **p_kwargs)
        setattr(args, 'd_model', d_model_val)
        sampled['d_model'] = d_model_val

    # Sample d_ff while enforcing d_ff >= d_model
    if 'd_ff' in parameters:
        p_kwargs = parameters['d_ff']['kwargs']
        p_type = parameters['d_ff']['type']
        if 'd_model' in sampled and p_type == 'categorical':
            valid_choices = [c for c in p_kwargs['choices'] if c >= sampled['d_model']]
            if not valid_choices:
                valid_choices = [sampled['d_model']]
            d_ff_val = trial.suggest_categorical('d_ff', choices=valid_choices)
        elif p_type == 'categorical':
            d_ff_val = trial.suggest_categorical('d_ff', **p_kwargs)
        elif p_type == 'int':
            d_ff_val = trial.suggest_int('d_ff', **p_kwargs)
        else:
            d_ff_val = trial.suggest_float('d_ff', **p_kwargs)
        setattr(args, 'd_ff', d_ff_val)
        sampled['d_ff'] = d_ff_val

    # Sample remaining parameters
    for param, param_kwargs in parameters.items():
        if param in sampled:
            continue

        p_type = param_kwargs['type']
        if p_type == 'categorical':
            value = trial.suggest_categorical(param, **param_kwargs['kwargs'])
        elif p_type == 'float':
            value = trial.suggest_float(param, **param_kwargs['kwargs'])
        elif p_type == 'int':
            value = trial.suggest_int(param, **param_kwargs['kwargs'])
        else:
            raise ValueError(f"Unknown parameter type: {p_type} for parameter {param}")
        setattr(args, param, value)

    # Maintain derivative attributes consistently during search
    if hasattr(args, 'd_model'):
        args.d_temp = args.d_model

    return args


def objective_func(trial, args, hp_configs, logger, base_seed=2021):
    # Isolated random seed per trial
    trial_seed = base_seed + trial.number
    set_seed(trial_seed)

    # Deepcopy args to avoid mutating the base args across trials
    trial_args = copy.deepcopy(args)

    subfolder = getattr(trial_args, 'results_subfolder', 'evaluation')
    setting = os.path.join(f'hp_search_{trial_args.model_id}_{trial_args.model}', f'trial_{trial.number}')

    try:
        # Collect suggested hyperparameters
        trial_args = suggest_params(trial, trial_args, hp_configs)
        exp = Exp_Long_Term_Forecast(trial_args)

        print(f"\n>>> Starting trial {trial.number} (seed={trial_seed}) <<<")
        exp.train(setting, trial)

        # Validate
        vali_data, vali_loader = exp._get_data(flag='val')
        criterion = exp._select_criterion()
        val_loss = exp.vali(vali_data, vali_loader, criterion)
        print(f"Trial {trial.number} validation loss: {val_loss:.6f}")

        return val_loss

    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} pruned")
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")

        error_dir = os.path.join('test_results', subfolder)
        os.makedirs(error_dir, exist_ok=True)

        # Log error details to the dynamic subfolder
        with open(os.path.join(error_dir, 'failed_trials.log'), 'a') as f:
            f.write(f"Trial {trial.number} failed:\nParameters: {trial.params}\nError: {str(e)}\n\n")
        return float('inf')

    finally:
        cleanup_path = os.path.join('./checkpoints/', setting)
        if os.path.exists(cleanup_path):
            shutil.rmtree(cleanup_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    logger = OutputLogger()

    hp_seed = 2021
    set_seed(hp_seed)

    parser = init_parser()
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    # Handle multi-GPU setup if needed
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Ensure 'is_training' is set to True
    args.is_training = 1

    # Dynamically select subfolder (default: 'evaluation')
    subfolder = getattr(args, 'results_subfolder', 'evaluation')

    # Persistent SQLite file for later visualization
    study_folder = f"hp_search_{args.model_id}_{args.model}"
    db_dir = os.path.join('test_results', subfolder, study_folder)
    os.makedirs(db_dir, exist_ok=True)
    logger.activate_file_logging(db_dir)

    # check if args.path_to_hp_config is provided
    if args.path_to_hp_config is None:
        raise ValueError("Provide a valid path to the hyperparameter config file using --path_to_hp_config")

    # load the hp config file
    with open(args.path_to_hp_config, 'r') as f:
        hp_configs = json.load(f)

    # SuccessiveHalvingPruner configuration
    pruner = HyperbandPruner(
        min_resource=hp_configs.get("min_resource", 3),
        reduction_factor=hp_configs.get("reduction_factor", 2),
        max_resource="auto"  # Optuna infers max epochs automatically from trial.report()
    )

    storage_url = f"sqlite:///{os.path.join(db_dir, 'optuna_study.db')}"
    study_name = f"hp_search_{args.model_id}_{args.model}"

    # create study instance
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='minimize',
        pruner=pruner,
        load_if_exists=True
    )

    bound_callback = partial(save_trials_callback, args=args)
    objective = partial(objective_func, args=args, hp_configs=hp_configs, logger=logger, base_seed=hp_seed)
    study.optimize(objective, n_trials=hp_configs["n_trials"], callbacks=[bound_callback])

    # Output search results
    print('\n' + '=' * 50)
    print('Hyperparameter Search Finished')
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value (Validation Loss):', trial.value)
    print('  Best Parameters:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    print('=' * 50 + '\n')

    # Save best parameters to JSON file for downstream shell scripts
    best_params_file = os.path.join(db_dir, f'best_params_{args.model_id}_{args.model}_pl{args.pred_len}.json')
    best_config_out = {
        "model_id": args.model_id,
        "model": args.model,
        "pred_len": args.pred_len,
        "best_val_loss": trial.value,
        "best_params": trial.params
    }
    with open(best_params_file, 'w') as f:
        json.dump(best_config_out, f, indent=4)

    print(f"Saved best parameter configuration to: {best_params_file}")

    logger.deactivate_file_logging()