from datasets import get_min_max_values, get_quantization_thresholds, load_data
import torch
import pandas as pd
import random
from tqdm import tqdm
from training import train_mlp_model, train_mlp_pre_model, train_soft_mlp, train_soft_comp_mlp
import argparse
import os
import numpy as np
import openml
from os.path import join

def train_and_evaluate_different_bits(df, loss_columns,
                                    train_loader, test_loader, 
                                    result_folder, dataset,  
                                    n_bits_min = 2, n_bits_max = 8,
                                    num_features=8, device = 'cpu'):
    
    min_values, max_values = get_min_max_values(train_loader, num_features=num_features)
    
    result_dict = {'bits': []}
    for col in loss_columns:
        result_dict[col] = []
    for n_bits in tqdm(range(n_bits_min, n_bits_max+1)):
        result_dict['bits'].append(n_bits)
        thresholds = get_quantization_thresholds(train_loader, n_bits)
        
        for column in loss_columns:
            df.sort_values(by=column, ascending=True, inplace=True)
            best_weight_decay = df['weight_decay'].values[0]
            best_learning_rate = df['learning_rate'].values[0]
            best_hidden_layers = df['hidden_layers'].values[0]
            best_hidden_neurons = df['hidden_neurons'].values[0]
            best_num_epochs = df['num_epochs'].values[0]
            best_decrease_factor = df['decrease_factor'].values[0]
            if column in ['val_loss_mlp',
                                    'val_loss_hard_post_mlp',
                                    'val_loss_hard_thr_post_mlp']:
                architecture = [num_features] + [best_hidden_neurons]*best_hidden_layers + [1]
                val_loss_mlp, val_loss_hard_post_mlp, val_loss_hard_thr_post_mlp = train_mlp_model(train_loader=train_loader, val_loader=test_loader, 
                            architecture=architecture,
                            min_values=min_values, max_values=max_values, thresholds=thresholds,
                            num_epochs=int(best_num_epochs), learning_rate=best_learning_rate,
                            weight_decay=best_weight_decay, add_noise=False, n_bits=n_bits, device=device)
                if column == 'val_loss_mlp':
                    result_dict['val_loss_mlp'].append(val_loss_mlp)
                elif column == 'val_loss_hard_post_mlp':
                    result_dict['val_loss_hard_post_mlp'].append(val_loss_hard_post_mlp)
                elif column == 'val_loss_hard_thr_post_mlp':
                    result_dict['val_loss_hard_thr_post_mlp'].append(val_loss_hard_thr_post_mlp)
            elif column in ['val_loss_hard_pre_mlp',
                            'val_loss_hard_thr_pre_mlp']:
                architecture = [num_features] + [best_hidden_neurons]*best_hidden_layers + [1]
                val_loss_hard_pre_mlp, val_loss_hard_thr_pre_mlp = train_mlp_pre_model(train_loader=train_loader, val_loader=test_loader,
                            architecture=architecture,
                            min_values=min_values, max_values=max_values, thresholds=thresholds,
                            num_epochs=int(best_num_epochs), learning_rate=best_learning_rate,
                            weight_decay=best_weight_decay, add_noise=False, n_bits=n_bits, device=device)
                if column == 'val_loss_hard_pre_mlp':
                    result_dict['val_loss_hard_pre_mlp'].append(val_loss_hard_pre_mlp)
                elif column == 'val_loss_hard_thr_pre_mlp':
                    result_dict['val_loss_hard_thr_pre_mlp'].append(val_loss_hard_thr_pre_mlp)
            elif column in ['val_loss_soft_mlp']:
                            # ,'val_loss_soft_hard_mlp']:
                architecture = [num_features] + [best_hidden_neurons]*best_hidden_layers + [1]
                val_loss_soft_mlp, val_loss_soft_hard_mlp = train_soft_mlp(train_loader=train_loader, val_loader=test_loader,
                            architecture=architecture,
                            min_values=min_values, max_values=max_values, thresholds=thresholds,
                            num_epochs=int(best_num_epochs), learning_rate=best_learning_rate,
                            weight_decay=best_weight_decay, add_noise=False, n_bits=n_bits,
                            decrease_factor=best_decrease_factor, device=device)
                result_dict['val_loss_soft_hard_mlp'].append(val_loss_soft_hard_mlp)   
                if column == 'val_loss_soft_mlp':
                    result_dict['val_loss_soft_mlp'].append(val_loss_soft_mlp)
                elif column == 'val_loss_soft_hard_mlp':
                    result_dict['val_loss_soft_hard_mlp'].append(val_loss_soft_hard_mlp)
            elif column in ['val_loss_soft_comp_mlp']:
                            # 'val_loss_soft_hard_comp_mlp']:
                n_thresholds_per_feature = 2 ** n_bits - 1
                architecture = [num_features * n_thresholds_per_feature] + [best_hidden_neurons]*best_hidden_layers + [1]
                val_loss_soft_comp_mlp, val_loss_soft_hard_comp_mlp = train_soft_comp_mlp(train_loader=train_loader, val_loader=test_loader,
                            architecture=architecture,
                            min_values=min_values, max_values=max_values, thresholds=thresholds,
                            num_epochs=int(best_num_epochs), learning_rate=best_learning_rate,
                            weight_decay=best_weight_decay, add_noise=False, n_bits=n_bits,
                            decrease_factor=best_decrease_factor, device=device)
                result_dict['val_loss_soft_hard_comp_mlp'].append(val_loss_soft_hard_comp_mlp)
                if column == 'val_loss_soft_comp_mlp':
                    result_dict['val_loss_soft_comp_mlp'].append(val_loss_soft_comp_mlp)
                elif column == 'val_loss_soft_hard_comp_mlp':
                    result_dict['val_loss_soft_hard_comp_mlp'].append(val_loss_soft_hard_comp_mlp)
        results_df = pd.DataFrame(result_dict) 
        results_df.to_csv(join(result_folder, f'{dataset}_evaluation_from_{n_bits_min}_to_{n_bits}bits.csv'), index=False)
        if n_bits > n_bits_min:
            os.remove(join(result_folder, f'{dataset}_evaluation_from_{n_bits_min}_to_{n_bits-1}bits.csv'))
    return results_df


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
    n_bits_hyperparameter = 4
    n_steps = 100
    if csv_folder is None:
        csv_folder = result_folder
    csv_path = join(csv_folder, f'{dataset}_hyperparameter_tuning_{n_bits_hyperparameter}bits_{n_steps}steps.csv')

    train_loader, val_loader, test_loader = load_data(dataset, scratch)
    for X, _ in train_loader:
        num_features = X.shape[1]
        break

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(csv_path)
    loss_columns = [col for col in df.columns if 'loss' in col]
    
    print(f"Evaluating best model from {n_bits_min}-bits quantized inputs to {n_bits_max} bits on dataset {dataset}")
    results_df_all = train_and_evaluate_different_bits(df=df, loss_columns=loss_columns,
                                                       train_loader=train_loader, test_loader=test_loader,
                                                        result_folder=result_folder,dataset=dataset,
                                                        n_bits_min=n_bits_min, n_bits_max=n_bits_max,
                                                        device=device,num_features=num_features)
    print(f"Finished random search for {n_steps} steps with {n_bits_max} bits on dataset {dataset}")
