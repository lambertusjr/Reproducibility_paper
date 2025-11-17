import optuna
from optuna.trial import Trial
import numpy as np
from models import GCN, ModelWrapper
import torch
import torch.nn as nn
from Helper_functions import calculate_metrics, balanced_class_weights, FocalLoss, _get_model_instance, check_study_existence, run_trial_with_cleanup, train_and_test_NMW_models, _run_wrapper_model_test
from tqdm import tqdm, trange
from training_and_testing import train_and_validate, train_and_test
from torch.optim import Adam

models = ['MLP', 'SVM', 'XGB', 'RF', 'GCN', 'GAT', 'GIN']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_EARLY_STOP = {
    "patience": 20,
    "min_delta": 1e-3,
}
EARLY_STOP_LOGGING = False


def _early_stop_args_from(source: dict) -> dict:
    """Build early stopping kwargs, falling back to defaults when keys are absent."""
    return {
        "patience": source.get("early_stop_patience", DEFAULT_EARLY_STOP["patience"]),
        "min_delta": source.get("early_stop_min_delta", DEFAULT_EARLY_STOP["min_delta"]),
        "log_early_stop": EARLY_STOP_LOGGING,
    }
    
def objective(trial, model, data, train_perf_eval, val_perf_eval, train_mask, val_mask):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Shared Hyperparameters ---
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.05, log=False)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-3, log=True)
    
    alpha_weights = balanced_class_weights(data.y[train_perf_eval])
    #loss_type = trial.suggest_categorical('loss_type', ['focal', 'cross_entropy'])
    loss_type = 'focal'
    if loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=alpha_weights.to(device))
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01, log=False) #Previous learning rate range was too high for CE
    else:
        gamma_focal = trial.suggest_float('gamma_focal', 0.2, 5.0)
        alpha_weights = alpha_weights.to(device)
        criterion = FocalLoss(alpha=alpha_weights, gamma=gamma_focal)
        
    early_stop_patience = trial.suggest_int('early_stop_patience', 5, 40)
    early_stop_min_delta = trial.suggest_float('early_stop_min_delta', 1e-4, 5e-3, log=True)
    num_epochs = trial.suggest_int('num_epochs', 50, 250)
    trial_early_stop_args = _early_stop_args_from({
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta
    })

    # --- Model Instantiation (Refactored) ---
    model_instance = _get_model_instance(trial, model, data, device)

    # --- Training and Evaluation ---
    wrapper_models = ['MLP', 'GCN', 'GAT', 'GIN']
    sklearn_models = ['SVM', 'XGB', 'RF']
    
    data = data.to(device)
    train_perf_eval = train_perf_eval.to(device)
    val_perf_eval = val_perf_eval.to(device)

    if model in wrapper_models:
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_wrapper = ModelWrapper(model_instance, optimizer, criterion)
        model_wrapper.model.to(device)
        
        metrics, best_model_wts, best_f1 = train_and_validate(
            model_wrapper, data,num_epochs=num_epochs,
            **trial_early_stop_args
        )
        return best_f1

    elif model in sklearn_models:
        model_instance.fit(data.x[train_perf_eval].cpu().numpy(), data.y[train_perf_eval].cpu().numpy())
        pred = model_instance.predict(data.x[val_perf_eval].cpu().numpy())
        prob = model_instance.predict_proba(data.x[val_perf_eval].cpu().numpy())
        metrics = calculate_metrics(data.y[val_perf_eval].cpu().numpy(), pred, prob)
        return metrics['f1_illicit']
    
    # All other model types (XGBe+GIN, GINe+XGB) have been removed.


def run_optimization(models, data, train_perf_eval, val_perf_eval, test_perf_eval, train_mask, val_mask, data_for_optimization):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Dynamically create result dictionaries ---
    model_parameters = {model_name: [] for model_name in models}
    
    METRIC_KEYS = [
        'accuracy', 'precision', 'precision_illicit', 'recall', 'recall_illicit',
        'f1', 'f1_illicit', 'roc_auc', 'roc_auc_illicit', 'PR_curve', 'PRAUC', 'kappa'
    ]
    testing_results = {
        model_name: {key: [] for key in METRIC_KEYS} 
        for model_name in models
    }
    
    focal_alpha = balanced_class_weights(data.y[train_perf_eval])
    focal_alpha_device = focal_alpha.to(device)

    for model_name in tqdm(models, desc="Models", unit="model"):
        sklearn_models = ['SVM', 'XGB', 'RF']
        if model_name in sklearn_models:
            n_trials = 70
        elif model_name == "MLP":
            n_trials = 100
        else:
            n_trials = 200
        study_name = f'{model_name}_optimization on {data_for_optimization} dataset'
        db_path = 'sqlite:///optimization_results.db'

        if check_study_existence(model_name, data_for_optimization): # Assumes check_study_existence logic is correct
            study = optuna.load_study(study_name=study_name, storage=db_path)
        else:
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=db_path,
                load_if_exists=True
            )
            with tqdm(total=n_trials, desc=f"{model_name} trials", leave=False, unit="trial") as trial_bar:
                def _optuna_progress_callback(study, trial):
                    trial_bar.update()
                study.optimize(
                    lambda trial: run_trial_with_cleanup( # Assumes run_trial_with_cleanup exists
                        objective, model_name, trial, model_name, data, train_perf_eval, val_perf_eval, train_mask, val_mask
                    ),
                    n_trials=n_trials,
                    callbacks=[_optuna_progress_callback]
                )

        print(f"Best hyperparameters for {model_name}:", study.best_params)
        model_parameters[model_name].append(study.best_params)
        params_for_model = study.best_params

        # --- Setup for final test runs ---
        loss_type = params_for_model.get("loss_type", "focal")
        if loss_type == "cross_entropy":
            criterion = nn.CrossEntropyLoss(weight=focal_alpha_device)
        else:
            gamma_focal = params_for_model.get("gamma_focal", 2.0)
            criterion = FocalLoss(alpha=focal_alpha, gamma=gamma_focal)
        early_stop_args = _early_stop_args_from(params_for_model)

        # --- Final Test Runs (Refactored) ---
        for _ in trange(30, desc=f"Runs for {model_name}", leave=False, unit="run"):
            test_metrics = {}
            best_f1 = None # Not all test runs return this

            match model_name:
                case "MLP" | "GCN" | "GAT" | "GIN":
                    test_metrics, best_f1 = _run_wrapper_model_test(
                        model_name, data, params_for_model, criterion, early_stop_args, device,
                        train_perf_eval, val_perf_eval, test_perf_eval
                    )
                case "SVM" | "RF" | "XGB":
                    test_metrics = train_and_test_NMW_models(
                        model_name=model_name,
                        data=data,
                        train_perf_eval=train_perf_eval,
                        val_perf_eval=val_perf_eval,
                        test_perf_eval=test_perf_eval,
                        params_for_model=params_for_model
                    )
                case _:
                    # This will now catch any models you removed, like 'XGBe+GIN'
                    print(f"Warning: No test logic defined for {model_name}. Skipping.")
                    continue
            
            # --- Centralized Metric Appending ---
            for key in METRIC_KEYS:
                if key in test_metrics:
                    testing_results[model_name][key].append(test_metrics[key])

    return model_parameters, testing_results