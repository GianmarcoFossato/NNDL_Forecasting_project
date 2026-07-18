import os
import torch
import optuna
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
from run import init_parser, OutputLogger
from optuna.pruners import SuccessiveHalvingPruner
from functools import partial
import json
import shutil

def save_trials_callback(study, trial, args):
    model_dir = f'test_results/{args.model_id}_{args.model}'
    os.makedirs(model_dir, exist_ok=True)

    df = study.trials_dataframe()
    last_trial_file = f'{model_dir}/trials_{args.model}_{args.model_id}.csv'
    df.to_csv(last_trial_file, index=False)

def suggest_params(trial, args, hp_configs):
    """Use Optuna to suggest hyperparameters based on the config file.
       Catch special cases for certain models if needed."""
    # Get the hyperparameter ranges from the config file
    parameters = hp_configs['parameters']

    # Override parameters for special cases
    special_cases = hp_configs['special_cases']
    if args.model in special_cases.keys():
        special_parameters = special_cases[args.model]['parameters']
        for param, special_values in special_parameters.items():
            parameters[param] = special_values

    # Get suggested values by Optuna
    for param, param_kwargs in parameters.items():
        if param_kwargs['type'] == 'categorical':
            value = trial.suggest_categorical(param, **param_kwargs['kwargs'])
        elif param_kwargs['type'] == 'float':
            value = trial.suggest_float(param, **param_kwargs['kwargs'])
        elif param_kwargs['type'] == 'int':
            value = trial.suggest_int(param, **param_kwargs['kwargs'])
        else:
            raise ValueError(f"Unknown parameter type: {param_kwargs['type']} for parameter {param}")
        args.__setattr__(param, value)


    ### KEEP AS REFERENCE
    # Special settings for ModernTCN due to its unique architecture
    # if args.model == "ModernTCN":
    #     args.num_blocks = torch.ones(args.e_layers, dtype=torch.int32).tolist()
    #     args.large_size = torch.ones(args.e_layers, dtype=torch.int32).tolist() * 51
    #     args.small_size = torch.ones(args.e_layers, dtype=torch.int32).tolist() * 5
    #     args.dims = torch.ones(args.e_layers, dtype=torch.int32).tolist() * args.d_model
    #     args.dw_dims = torch.ones(args.e_layers, dtype=torch.int32).tolist() * args.d_model

    # Fix the feedforward dimension to be equal to d_model for the HP search
    args.d_ff = args.d_model
    return args


def objective_func(trial, args, hp_configs, logger):
    setting = f'hp_search_{args.model_id}_{args.model}/trial_{trial.number}'

    try:
        # collect suggested hyperparameters
        args = suggest_params(trial, args, hp_configs)
        exp = Exp_Long_Term_Forecast(args)
        #setting = f'hp_search_{args.model_id}_{args.model}/trial_{trial.number}'

        print(f">>>Starting trial {trial.number}<<<")
        exp.train(setting, trial)

        # Validate
        vali_data, vali_loader = exp._get_data(flag='val')
        criterion = exp._select_criterion()
        val_loss = exp.vali(vali_data, vali_loader, criterion)
        print(f"Trial {trial.number} validation loss: {val_loss}")

        # Clean up any leftover files/resources
        # setting = f'hp_search_{args.model_id}_{args.model}/trial_{trial.number}'
        # cleanup_path = os.path.join('./checkpoints/', setting)
        # if os.path.exists(cleanup_path):
        #     shutil.rmtree(cleanup_path)

        return val_loss

    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} pruned")
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")

        os.makedirs('test_results', exist_ok=True)

        # log error details
        with open('test_results/failed_trials.log', 'a') as f:
            f.write(f"Trial {trial.number} failed:\nParameters: {trial.params}\nError: {str(e)}\n\n")
        return float('inf')

    finally:
        cleanup_path = os.path.join('./checkpoints/', setting)
        if os.path.exists(cleanup_path):
            shutil.rmtree(cleanup_path)


def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    logger = OutputLogger()

    hp_seed = 2021
    test_seed = hp_seed
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

    # Global to track last file
    last_trial_file = None

    # Persistent SQLite file for later visualization
    study_folder = f"{args.model_id}_{args.model}"
    db_dir = os.path.join('test_results', study_folder)
    os.makedirs(db_dir, exist_ok=True)
    logger.activate_file_logging(db_dir)

    # check if args.path_to_hp_config is provided
    if args.path_to_hp_config is None:
        raise ValueError("Provide a valid path to the hyperparameter config file using --path_to_hp_config")

    # load the hp config file
    with open(args.path_to_hp_config, 'r') as f:
        hp_configs = json.load(f)

    # initialize Optuna study with SuccessiveHalvingPruner
    pruner = SuccessiveHalvingPruner(
        min_resource=hp_configs["min_resource"],  # Minimum number of epochs
        reduction_factor=hp_configs["reduction_factor"],  # Factor to reduce the number of trials
        min_early_stopping_rate=hp_configs["min_early_stopping_rate"]
    )

    storage_url = f"sqlite:///{os.path.join(db_dir, 'optuna_study.db')}"
    study_name = f"hp_search_{args.model_id}_{args.model}"

    # create study instance
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='minimize',
        pruner=pruner,
        load_if_exists=True  # Allows resuming optimization
    )

    # pass logger instance down to the objective callback
    bound_callback = partial(save_trials_callback, args=args)
    objective = partial(objective_func, args=args, hp_configs=hp_configs, logger=logger)
    study.optimize(objective, n_trials=hp_configs["n_trials"], callbacks=[bound_callback])

    # output the best hyperparameters
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value:', trial.value)

    for key, value in trial.params.items():
        setattr(args, key, value)

    args.d_ff = args.d_model
    args.d_temp = args.d_model
    exp = Exp_Long_Term_Forecast(args)

    for ii in range(args.itr):
        test_seed += 1
        set_seed(test_seed)
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii, test_seed)
        exp.train(setting)

        # Test the model
        exp.test(setting)

    logger.deactivate_file_logging()