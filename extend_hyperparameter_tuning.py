from datasets import get_min_max_values, get_quantization_thresholds, load_data, get_minmax_thresholds
import torch
import pandas as pd
import random
from tqdm import tqdm
from training import train_mlp_model, train_mlp_pre_model, train_soft_mlp, train_soft_comp_mlp, train_hard_comp_mlp
import argparse
import os
import numpy as np
from os.path import join

def extend_hyperparameter_tuning(csv_folder, 
                                train_loader, test_loader, 
                                result_folder, dataset,  
                                n_bits_min = 2, n_bits_max = 8, n_steps = 100,
                                num_features=8, device = 'cpu'):
     
    min_values, max_values = get_min_max_values(train_loader, num_features=num_features)
    
    result_dict = {'bits': []}
    for col in ['val_loss_hard_bitwise_minmax_mlp', 'val_loss_hard_bitwise_quantile_mlp']:
        result_dict[col] = []
    for n_bits in tqdm(range(n_bits_min, n_bits_max+1)):
        result_dict['bits'].append(n_bits)
        thresholds = get_quantization_thresholds(train_loader, n_bits)
        minmax_thresholds = get_minmax_thresholds(min_values, max_values, n_bits)
        csv_path = join(csv_folder, f'{dataset}_hyperparameter_tuning_{n_bits}bits_{n_steps}steps.csv')
        df = pd.read_csv(csv_path)
        df['val_loss_hard_bitwise_minmax_mlp'] = np.nan
        df['val_loss_hard_bitwise_quantile_mlp'] = np.nan

        for row in range(len(df)):
            df_row = df.iloc[row]
            weight_decay = df_row['weight_decay']
            learning_rate = df_row['learning_rate']
            hidden_layers = df_row['hidden_layers'].astype(int)
            hidden_neurons = df_row['hidden_neurons'].astype(int)
            num_epochs = df_row['num_epochs'].astype(int)
            decrease_factor = df_row['decrease_factor']
            architecture = [num_features] + [hidden_neurons] * hidden_layers + [1]

            val_loss_hard_bitwise_minmax_mlp, val_loss_hard_bitwise_quantile_mlp = train_hard_comp_mlp(train_loader=train_loader, val_loader=test_loader, 
                                                         architecture=architecture, minmax_thresholds=minmax_thresholds, thresholds=thresholds,
                                                         num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                                                         n_bits=n_bits, decrease_factor=decrease_factor, device=device)
            df.at[row,'val_loss_hard_bitwise_minmax_mlp'] = val_loss_hard_bitwise_minmax_mlp
            df.at[row,'val_loss_hard_bitwise_quantile_mlp'] = val_loss_hard_bitwise_quantile_mlp
            
            df.to_csv(join(result_folder, f'{dataset}_hyperparameter_tuning_{n_bits}bits_{n_steps}steps_extended{row+1}.csv'), index=False)
            if row > 0:
                os.remove(join(result_folder, f'{dataset}_hyperparameter_tuning_{n_bits}bits_{n_steps}steps_extended{row}.csv'))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    # Add --dataset, --type, and --train argument
    parser.add_argument('--dataset', type=str, required=True,
                        help='Number of iterations')

    parser.add_argument('--scratch', type=str, required=True,
                        help='Number of iterations')
    parser.add_argument('--n_steps', type=int, default=1,
                        help='Number of iterations')
    parser.add_argument('--result_folder', type=str, default='results',
                        help='Folder to save results')
    parser.add_argument('--csv_folder', type=str, 
                        default=None,required=False)
    parser.add_argument('--n_bits_min', type=int, default=2, required=False,
                        help='Minimum number of bits for quantization')
    parser.add_argument('--n_bits_max', type=int, default=8, required=False,
                        help='Maximum number of bits for quantization')
    args = parser.parse_args()
    dataset = args.dataset
    scratch = args.scratch
    n_steps = args.n_steps
    result_folder = args.result_folder
    csv_folder = args.csv_folder
    n_bits_min = args.n_bits_min
    n_bits_max = args.n_bits_max
    if csv_folder is None:
        csv_folder = result_folder

    train_loader, val_loader, test_loader = load_data(dataset, scratch)
    for X, _ in train_loader:
        num_features = X.shape[1]
        break

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    print(f"Extend hyperparameter tuning for {n_steps} steps from {n_bits_min} bits to {n_bits_max} bits on dataset {dataset}")
    results_df_all = extend_hyperparameter_tuning(csv_folder=csv_folder,
                                                    train_loader=train_loader, test_loader=test_loader,
                                                    result_folder=result_folder,dataset=dataset,
                                                    n_bits_min=n_bits_min, n_bits_max=n_bits_max,
                                                    device=device,num_features=num_features)
    print(f"Extended hyperparameter tuning for {dataset} completed.")
