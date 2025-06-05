from datasets import get_min_max_values, get_quantization_thresholds, load_data, get_minmax_thresholds
import torch
import pandas as pd
import random
from tqdm import tqdm
from training import train_mlp_model, train_mlp_pre_model, train_soft_mlp, train_soft_comp_mlp, train_hard_comp_mlp
import argparse
import os
import numpy as np
import openml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

def random_search_cv(X_tensor : torch.tensor, y_tensor : torch.tensor, result_folder, dataset,
                                              n_steps = 10, n_bits =8, optimize_dict = {}, device = 'cpu', debug=False):
    num_features = X_tensor.shape[1]
    # Define default hyperparameters
    weight_decay =  0
    learning_rate = 0.001
    hidden_layers = 2
    hidden_neurons = 256
    num_epochs = 30
    add_noise = False
    decrease_factor = 0.001

    # Lists to store results
    val_loss_mlp_values = []
    val_loss_hard_post_mlp_values = []
    val_loss_hard_thr_post_mlp_values = []
    val_loss_hard_pre_mlp_values = []
    val_loss_hard_thr_pre_mlp_values = []
    val_loss_soft_mlp_values = []
    val_loss_soft_hard_mlp_values = []
    val_loss_soft_comp_mlp_values = []
    val_loss_soft_hard_comp_mlp_values = []
    val_loss_hard_bitwise_minmax_mlp_values = []
    val_loss_hard_bitwise_quantile_mlp_values = []

    train_loss_mlp_values = []
    train_loss_hard_pre_mlp_values = []
    train_loss_hard_thr_pre_mlp_values = []
    train_loss_soft_mlp_values = []
    train_loss_soft_comp_mlp_values = []
    train_loss_hard_bitwise_minmax_mlp_values = []
    train_loss_hard_bitwise_quantile_mlp_values = []
    kFold_id = []
    hyperparameter_setting_id = []
    hyperparameter_dict = {
        'weight_decay': [],
        'learning_rate': [],
        'hidden_layers': [],
        'hidden_neurons': [],
        'num_epochs': [],
        'decrease_factor': []}

    results_df = pd.DataFrame()
    # Perform random search
    for f in tqdm(range(n_steps)):
        for key, value in optimize_dict.items():
            if key == 'weight_decay':
                weight_decay = random.choice(value)
            elif key == 'learning_rate':
                learning_rate = random.choice(value)
            elif key == 'hidden_layers':
                hidden_layers = random.choice(value)
            elif key == 'hidden_neurons':
                hidden_neurons = random.choice(value)
                if debug:
                    hidden_neurons = 10
            elif key == 'num_epochs':
                num_epochs = random.choice(value)    
            elif key == 'add_noise':
                add_noise = random.choice(value)    
            elif key == 'decrease_factor':
                decrease_factor = random.choice(value)
            else:
                raise ValueError(f"Unknown hyperparameter: {key}")
        k_folds = 5
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        counter = 0
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tensor)):
            hyperparameter_setting_id.append(f)
            kFold_id.append(counter)
            counter += 1
            X_train_tensor = X_tensor[train_idx]
            y_train_tensor = y_tensor[train_idx]
            X_val_tensor = X_tensor[val_idx]
            y_val_tensor = y_tensor[val_idx]

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_train_array = scaler_X.fit_transform(X_train_tensor)
            y_train_array = scaler_y.fit_transform(y_train_tensor)
            X_val_array = scaler_X.transform(X_val_tensor)
            y_val_array = scaler_y.transform(y_val_tensor)

            X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_array, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_array, dtype=torch.float32)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            thresholds = get_quantization_thresholds(train_loader, n_bits)
            min_values, max_values = get_min_max_values(train_loader, num_features=num_features)
            minmax_thresholds = get_minmax_thresholds(min_values, max_values, n_bits)

            architecture = [num_features] + [hidden_neurons] * hidden_layers + [1]
            hyperparameter_dict['weight_decay'].append(weight_decay)
            hyperparameter_dict['learning_rate'].append(learning_rate)
            hyperparameter_dict['hidden_layers'].append(hidden_layers)
            hyperparameter_dict['hidden_neurons'].append(hidden_neurons)
            hyperparameter_dict['num_epochs'].append(num_epochs)
            hyperparameter_dict['decrease_factor'].append(decrease_factor)

            # Calculate losses for mlp model
            val_loss_mlp, val_loss_hard_post_mlp, val_loss_hard_thr_post_mlp, train_loss_mlp = train_mlp_model(train_loader=train_loader, val_loader=val_loader,
                architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
                num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                n_bits=n_bits, device=device)

            # Calculate losses for pre-training quantization model
            val_loss_hard_pre_mlp, val_loss_hard_thr_pre_mlp, train_loss_hard_thr_pre_mlp_mm, train_loss_hard_thr_pre_mlp_q = train_mlp_pre_model(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
                num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                n_bits=n_bits, device=device)

            # Calculate losses for quantization model
            val_loss_soft_mlp, val_loss_soft_hard_mlp, train_loss_soft = train_soft_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
                num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                n_bits=n_bits, decrease_factor=decrease_factor, device=device)

            # Calculate losses for quantization model
            val_loss_soft_comp_mlp, val_loss_soft_hard_comp_mlp, train_loss_soft_hard_comp = train_soft_comp_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
                num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                n_bits=n_bits, decrease_factor=decrease_factor, device=device)

            # Calculate losses for hard quantization model
            val_loss_hard_bitwise_minmax_mlp, val_loss_hard_bitwise_quantile_mlp, train_loss_hard_bitwise_mm, train_loss_hard_bitwise_q = train_hard_comp_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, minmax_thresholds=minmax_thresholds, thresholds=thresholds,
                                                        num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                                                        n_bits=n_bits, decrease_factor=decrease_factor, device=device)

            val_loss_mlp_values.append(val_loss_mlp)
            val_loss_hard_post_mlp_values.append(val_loss_hard_post_mlp)
            val_loss_hard_thr_post_mlp_values.append(val_loss_hard_thr_post_mlp)
            val_loss_hard_pre_mlp_values.append(val_loss_hard_pre_mlp)
            val_loss_hard_thr_pre_mlp_values.append(val_loss_hard_thr_pre_mlp)
            val_loss_soft_mlp_values.append(val_loss_soft_mlp)
            val_loss_soft_hard_mlp_values.append(val_loss_soft_hard_mlp)
            val_loss_soft_comp_mlp_values.append(val_loss_soft_comp_mlp)
            val_loss_soft_hard_comp_mlp_values.append(val_loss_soft_hard_comp_mlp)
            val_loss_hard_bitwise_minmax_mlp_values.append(val_loss_hard_bitwise_minmax_mlp)
            val_loss_hard_bitwise_quantile_mlp_values.append(val_loss_hard_bitwise_quantile_mlp)

            train_loss_mlp_values.append(train_loss_mlp)
            train_loss_hard_pre_mlp_values.append(train_loss_hard_thr_pre_mlp_mm)
            train_loss_hard_thr_pre_mlp_values.append(train_loss_hard_thr_pre_mlp_q)
            train_loss_soft_mlp_values.append(train_loss_soft)
            train_loss_soft_comp_mlp_values.append(train_loss_soft_hard_comp)
            train_loss_hard_bitwise_minmax_mlp_values.append(train_loss_hard_bitwise_mm)
            train_loss_hard_bitwise_quantile_mlp_values.append(train_loss_hard_bitwise_q)

            losses_df = pd.DataFrame({
                'hyperparameter_setting_id' : hyperparameter_setting_id,
                'kFold_id': kFold_id,
                'val_loss_mlp': val_loss_mlp_values,
                'val_loss_hard_post_mlp': val_loss_hard_post_mlp_values,
                'val_loss_hard_thr_post_mlp': val_loss_hard_thr_post_mlp_values,
                'val_loss_hard_pre_mlp': val_loss_hard_pre_mlp_values,
                'val_loss_hard_thr_pre_mlp': val_loss_hard_thr_pre_mlp_values,
                'val_loss_soft_mlp': val_loss_soft_mlp_values,
                'val_loss_soft_hard_mlp': val_loss_soft_hard_mlp_values,
                'val_loss_soft_comp_mlp': val_loss_soft_comp_mlp_values,
                'val_loss_soft_hard_comp_mlp': val_loss_soft_hard_comp_mlp_values,
                'val_loss_hard_bitwise_minmax_mlp': val_loss_hard_bitwise_minmax_mlp_values,
                'val_loss_hard_bitwise_quantile_mlp': val_loss_hard_bitwise_quantile_mlp_values,
                'train_loss_mlp': train_loss_mlp_values,
                'train_loss_hard_pre_mlp': train_loss_hard_pre_mlp_values,
                'train_loss_hard_thr_pre_mlp': train_loss_hard_thr_pre_mlp_values,
                'train_loss_soft_mlp': train_loss_soft_mlp_values,
                'train_loss_soft_comp_mlp': train_loss_soft_comp_mlp_values,
                'train_loss_hard_bitwise_minmax_mlp': train_loss_hard_bitwise_minmax_mlp_values,
                'train_loss_hard_bitwise_quantile_mlp': train_loss_hard_bitwise_quantile_mlp_values
            })

            # Create DataFrame with results
            results_df = pd.DataFrame(hyperparameter_dict)
            results_df = pd.concat([results_df, losses_df], axis=1)

            results_df = results_df.sort_values(['hyperparameter_setting_id', 'val_loss_mlp'])  # Sort by loss ascending
            folder = Path(f'{result_folder}')
            folder.mkdir(parents=True, exist_ok=True)

            results_df.to_csv(f'{result_folder}/{dataset}_hyperparameter_tuning_{n_bits}bits_{f+1}steps.csv', index=False)
            # Delete the intermediate CSV file
        if f > 0:
            os.remove(f'{result_folder}/{dataset}_hyperparameter_tuning_{n_bits}bits_{f}steps.csv')

    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    # Add --dataset, --type, and --train argument
    parser.add_argument('--dataset', type=str, required=True,
                        help='Number of iterations')

    parser.add_argument('--scratch', type=str, required=True,
                        help='Number of iterations')
    parser.add_argument('--n_steps', type=int, default=2,
                        help='Number of iterations')
    parser.add_argument('--n_bits', type=int, default=4,
                        help='Number of bits for quantization')
    parser.add_argument('--result_folder', type=str, default='results',
                        help='Folder to save results')
    parser.add_argument('--debug', action='store_true', help='Should be in debug modus?')
    parser.add_argument('--no-debug', action='store_false', dest='debug', help='Do not start in debug modus')
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    dataset = args.dataset
    scratch = args.scratch
    n_steps = args.n_steps
    n_bits = args.n_bits
    debug = args.debug
    result_folder = args.result_folder

    X_tensor, y_tensor = load_data(dataset, scratch, False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimize_dict = {'weight_decay': [0, 0.0001],
                    'learning_rate': [0.001, 0.0001],
                    'hidden_layers': [5,6,8,10],
                    'hidden_neurons': [64,128,256,512,1024, 2048, 4096, 8192],
                    'num_epochs': [30,50,70],
                    'decrease_factor': [0.001, 0.0001]}
    
    print(f"Running random search for {n_steps} steps with {n_bits} bits on dataset {dataset}")

    results_df_all = random_search_cv(X_tensor=X_tensor, y_tensor=y_tensor,
                                                            result_folder=result_folder,
                                                            dataset=dataset,
                                                            n_bits=n_bits,
                                                            n_steps=n_steps,
                                                            optimize_dict=optimize_dict,
                                                            device=device, debug=debug)
    print(f"Finished random search for {n_steps} steps with {n_bits} bits on dataset {dataset}")
