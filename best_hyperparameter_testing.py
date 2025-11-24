from datasets import get_min_max_values, get_quantization_thresholds, load_data, get_minmax_thresholds
import torch
import pandas as pd
import random
from tqdm import tqdm
from training import train_fp_model, train_pre_model, train_SQ,train_SQplus, train_BwSQ, train_BwMQ_BwQQ, train_llt, train_lsq
import argparse
import os
import re
import glob
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import numpy as np

def create_random_search_df(optimize_dict, k_folds, n_steps, n_bits):
    np.random.seed(42)
    hyperparameter_df = pd.DataFrame(columns = ['hyperparameter_setting_id','kFold_id','bits'] + list(optimize_dict.keys()), index = range(n_steps * k_folds))
    hyperparameter_df['hyperparameter_setting_id'] = np.repeat(np.arange(n_steps), k_folds)
    hyperparameter_df['kFold_id'] = np.tile(np.arange(k_folds), n_steps)
    hyperparameter_df['bits'] = n_bits
    for key, value_list in optimize_dict.items():
        value_array = np.array(value_list)
        sampled = np.random.choice(value_array, size=n_steps, replace=True)
        hyperparameter_df[key] = np.repeat(sampled, k_folds).tolist()
    return hyperparameter_df

def get_results(dataset, n_bits):
    file_path = f'results/{dataset}/{dataset}_hyperparameter_tuning_{n_bits}bits_400iterations.csv'
    df = pd.read_csv(file_path)
    return df



def best_hyperparameter_testing(dataset, X_tensor, y_tensor, result_folder, device = 'cpu'):
    num_features = X_tensor.shape[1]

    n_bits_min = 2
    n_bits_max = 8

    X_cv_array, X_test_array, y_cv_array, y_test_array = train_test_split(X_tensor.numpy(), y_tensor.numpy(), test_size=0.1, random_state=42)

    X_train_tensor = X_cv_array
    y_train_tensor = y_cv_array

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_array = scaler_X.fit_transform(X_train_tensor)
    y_train_array = scaler_y.fit_transform(y_train_tensor)
    X_test_array_scaled = scaler_X.transform(X_test_array)
    y_test_array_scaled = scaler_y.transform(y_test_array)

    X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_array_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_array_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)



    results_df = get_results(dataset, n_bits = 2)
    val_cols = [col for col in results_df.columns if 'val_loss_' in col]
    val_cols = [col for col in val_cols if not col.endswith('_train')]
    methods = [col.split('_')[2] for col in val_cols]

    test_cols = ['test_loss_' + method for method in methods]
    time_cols = ['time_' + method for method in methods]
    hyperparameter_setting_cols = ['best_hyperparameter_id_' + method for method in methods]
    df_test = pd.DataFrame(columns = hyperparameter_setting_cols + val_cols + test_cols + time_cols, index = range(n_bits_max - n_bits_min + 1))

    for n_bits in tqdm(range(n_bits_min, n_bits_max + 1)):
        SEED = 42 + n_bits
        # Set seeds for Python, NumPy, and PyTorch
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        quantile_thresholds = get_quantization_thresholds(train_loader, n_bits)
        min_values, max_values = get_min_max_values(train_loader, num_features=num_features)
        minmax_thresholds = get_minmax_thresholds(min_values, max_values, n_bits)

        results_df = get_results(dataset, n_bits=n_bits)
        df_hyperparameter_avg = results_df.groupby('hyperparameter_setting_id').mean()
        df_val_loss = df_hyperparameter_avg[val_cols]
        best_hyperparameter_ids = df_val_loss.idxmin()
        best_val_losses = df_val_loss.min()
        df_test.loc[n_bits, hyperparameter_setting_cols] = best_hyperparameter_ids.values
        df_test.loc[n_bits, val_cols] = best_val_losses.values

        for method in methods:
            best_hyperparameter_id = df_test.at[n_bits, 'best_hyperparameter_id_' + method]
            row = results_df[results_df['hyperparameter_setting_id'] == best_hyperparameter_id].iloc[0]
            weight_decay = row['weight_decay']
            learning_rate = row['learning_rate']
            hidden_layers = int(row['hidden_layers'])
            hidden_neurons = int(row['hidden_neurons'])
            num_epochs = int(row['num_epochs'])
            decrease_factor = row['decrease_factor']
            dropout_rate = row['dropout_rate']

            architecture = [num_features] + [hidden_neurons] * hidden_layers + [1]

            # Calculate losses for FP and Post-quantization models
            if method in ['FP', 'Po-MQ', 'Po-QQ']:
                val_loss_FP, val_loss_PoMQ, val_loss_PoQQ, test_loss_FP, test_loss_PoMQ, test_loss_PoQQ, train_loss_FP, time_FP = train_fp_model(train_loader=train_loader, val_loader=test_loader, test_loader=test_loader,
                    architecture=architecture, min_values=min_values, max_values=max_values, quantile_thresholds=quantile_thresholds, dropout=dropout_rate,
                    num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                    n_bits=n_bits, device=device)
                if method == 'FP':
                    df_test.at[n_bits, 'test_loss_FP'] = test_loss_FP
                elif method == 'Po-MQ':
                    df_test.at[n_bits, 'test_loss_Po-MQ'] = test_loss_PoMQ
                elif method == 'Po-QQ':
                    df_test.at[n_bits, 'test_loss_Po-QQ'] = test_loss_PoQQ
                df_test.at[n_bits, f'time_{method}'] = time_FP

            
            # Calculate losses for pre-training quantization model
            elif method in ['Pr-MQ', 'Pr-QQ']:
                val_loss_PrMQ, val_loss_PrQQ, test_loss_PrMQ, test_loss_PrQQ, train_loss_PrMQ, train_loss_PrQQ, time_PrMQ = train_pre_model(train_loader=train_loader, val_loader=test_loader, test_loader=test_loader, architecture=architecture,
                                                                                                            min_values=min_values, max_values=max_values, quantile_thresholds=quantile_thresholds,
                    num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout_rate,
                    n_bits=n_bits, device=device)
                if method == 'Pr-MQ':
                    df_test.at[n_bits, 'test_loss_Pr-MQ'] = test_loss_PrMQ
                elif method == 'Pr-QQ':
                    df_test.at[n_bits, 'test_loss_Pr-QQ'] = test_loss_PrQQ
                df_test.at[n_bits, f'time_{method}'] = time_PrMQ

            # Calculate losses for soft quantization model
            elif method == 'SQ':
                val_loss_SQ_train, val_loss_SQ_inf, test_loss_SQ_train, test_loss_SQ_inf, train_loss_SQ, time_SQ = train_SQ(train_loader=train_loader, val_loader=test_loader, test_loader=test_loader, architecture=architecture, min_values=min_values, max_values=max_values, quantile_thresholds=quantile_thresholds,
                    num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout_rate,
                    n_bits=n_bits, decrease_factor=decrease_factor, device=device)
                df_test.at[n_bits, 'test_loss_SQ'] = test_loss_SQ_inf
                df_test.at[n_bits, f'time_{method}'] = time_SQ
            
            # Calculate losses for soft quantization model
            elif method == 'SQplus':
                val_loss_SQPlus_train, val_loss_SQPlus_inf, test_loss_SQPlus_train, test_loss_SQPlus_inf, train_loss_SQPlus, time_SQPlus = train_SQplus(train_loader=train_loader, val_loader=test_loader, test_loader=test_loader, architecture=architecture, min_values=min_values, max_values=max_values, quantile_thresholds=quantile_thresholds,
                    num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout_rate,
                    n_bits=n_bits, decrease_factor=decrease_factor, device=device)
                df_test.at[n_bits, 'test_loss_SQplus'] = test_loss_SQPlus_inf
                df_test.at[n_bits, f'time_{method}'] = time_SQPlus

            # Calculate losses for LSQ quantization model
            elif method == 'LSQ':
                val_loss_LSQ, test_loss_LSQ,train_loss_LSQ, time_LSQ = train_lsq(train_loader=train_loader, val_loader=test_loader, test_loader=test_loader,
                                                                                    architecture=architecture, min_values=min_values, max_values=max_values,
                                                                                    quantile_thresholds=quantile_thresholds,
                                                                                    num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout_rate,
                                                                                    n_bits=n_bits, decrease_factor=decrease_factor, device=device)
                df_test.at[n_bits, 'test_loss_LSQ'] = test_loss_LSQ
                df_test.at[n_bits, f'time_{method}'] = time_LSQ

            elif method == 'LLT9' or method == 'LLT4':
                val_loss_llt9, val_loss_llt_training9, test_loss_llt9, test_loss_llt_training9, train_loss_llt9, time_llt9, \
            val_loss_llt4, val_loss_llt_training4, test_loss_llt4, test_loss_llt_training4, train_loss_llt4, time_llt4 = train_llt(train_loader=train_loader, val_loader=test_loader, test_loader=test_loader,
                                                                                    architecture=architecture, min_values=min_values, max_values=max_values,
                                                                                    quantile_thresholds=quantile_thresholds,
                                                                                    num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout_rate,
                                                                                    n_bits=n_bits, decrease_factor=decrease_factor, device=device)
                if method == 'LLT9':
                    df_test.at[n_bits, 'test_loss_LLT9'] = test_loss_llt9
                    df_test.at[n_bits, f'time_{method}'] = time_llt9
                elif method == 'LLT4':
                    df_test.at[n_bits, 'test_loss_LLT4'] = test_loss_llt4
                    df_test.at[n_bits, f'time_{method}'] = time_llt4    

            # Calculate losses for bitwise soft quantization model
            elif method == 'Bw-SQ':
                val_loss_BwSQ_train, val_loss_BwSQ_inf, test_loss_BwSQ_train, test_loss_BwSQ_inf, train_loss_BwSQ, time_BwSQ = train_BwSQ(train_loader=train_loader, val_loader=test_loader, test_loader=test_loader, architecture=architecture, min_values=min_values, max_values=max_values, quantile_thresholds=quantile_thresholds,
                    num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout_rate,
                    n_bits=n_bits, decrease_factor=decrease_factor, device=device)
                df_test.at[n_bits, 'test_loss_Bw-SQ'] = test_loss_BwSQ_inf
                df_test.at[n_bits, f'time_{method}'] = time_BwSQ


            # Calculate losses for scalar, resp. bitwise, minmax, resp. quantile, quantization model
            elif method in ['Bw-MQ', 'Bw-QQ']:
                val_loss_BwMQ, val_loss_BwQQ, test_loss_BwMQ, test_loss_BwQQ, train_loss_BwMQ, train_loss_BwQQ, time_BwMQ = train_BwMQ_BwQQ(train_loader=train_loader, val_loader=test_loader, test_loader=test_loader, architecture=architecture, minmax_thresholds=minmax_thresholds, quantile_thresholds=quantile_thresholds,
                                                        num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout_rate,
                                                        n_bits=n_bits, decrease_factor=decrease_factor, device=device)
                if method == 'Bw-MQ':
                    df_test.at[n_bits, 'test_loss_Bw-MQ'] = test_loss_BwMQ
                    df_test.at[n_bits, f'time_{method}'] = time_BwMQ
                elif method == 'Bw-QQ':
                    df_test.at[n_bits, 'test_loss_Bw-QQ'] = test_loss_BwQQ
                    df_test.at[n_bits, f'time_{method}'] = time_BwMQ    
    df_test.to_csv(f'{result_folder}/{dataset}/{dataset}_best_hyperparameter_test.csv', index=False)
    return df_test                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    # Add --dataset, --type, and --train argument
    # parser.add_argument('--dataset', type=str, required=True,
    #                     help='Number of iterations')

    parser.add_argument('--scratch', type=str, required=True,
                        help='directory to save the results to')
    parser.add_argument('--result_folder', type=str, default='results',
                        help='Folder to save results')

    args = parser.parse_args()
    # dataset = args.dataset
    scratch = args.scratch
    result_folder = args.result_folder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['sulfur', 'cpu_act', 'california', 'fried', 'superconduct']
    for dataset in datasets:
        print(f"Running best hyperparameter testing on dataset {dataset} on device {device}")
        X_tensor, y_tensor = load_data(dataset, scratch, False)

        results_df_all = best_hyperparameter_testing(X_tensor=X_tensor, y_tensor=y_tensor,
                                                    result_folder=result_folder,
                                                    dataset=dataset,
                                                    device=device)
        print(f"Finished best hyperparameter testing on dataset {dataset}")
        stop
