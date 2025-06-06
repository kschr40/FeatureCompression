from datasets import load_data, get_min_max_values, get_quantization_thresholds, get_minmax_thresholds
import torch
import pandas as pd
import random
from tqdm import tqdm
from training import train_mlp_model, train_mlp_pre_model, train_soft_mlp, train_soft_comp_mlp, train_hard_comp_mlp
import argparse
from os.path import join
import os

def train_and_evaluate_best_models_multiple_bits(csv_folder,
                                                dataset, result_folder,
                                                n_runs = 5, 
                                                n_bits_min = 2, n_bits_max = 8, 
                                                device = 'cuda'):
    train_loader, val_loader, test_loader = load_data(dataset, scratch)
    for X, _ in train_loader:
        num_features = X.shape[1]
        break    
    for n_bits in range(n_bits_min, n_bits_max + 1):
        csv_path = join(csv_folder, f'{dataset}_hyperparameter_tuning_{n_bits}bits_{n_steps}steps_extended{n_steps}.csv')
        df = pd.read_csv(csv_path)
        loss_columns = [col for col in df.columns if 'loss' in col]
        result_dict = {}
        for col in loss_columns:
            result_dict[col] = []

        for f in tqdm(range(n_runs)):
            train_loader, val_loader, test_loader = load_data(dataset, scratch)
            min_values, max_values = get_min_max_values(train_loader, num_features)
            thresholds = get_quantization_thresholds(train_loader, n_bits)
            minmax_thresholds = get_minmax_thresholds(min_values, max_values, n_bits)
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
                elif column in ['val_loss_hard_bitwise_minmax_mlp', 'val_loss_hard_bitwise_quantile_mlp']:
                    architecture = [num_features * n_thresholds_per_feature] + [best_hidden_neurons]*best_hidden_layers + [1]
                    val_loss_hard_bitwise_minmax_mlp, val_loss_hard_bitwise_quantile_mlp = train_hard_comp_mlp(train_loader=train_loader, val_loader=test_loader, 
                                                         architecture=architecture, minmax_thresholds=minmax_thresholds, thresholds=thresholds,
                                                         num_epochs=best_num_epochs, learning_rate=best_learning_rate, weight_decay=best_weight_decay,
                                                         n_bits=n_bits, decrease_factor=best_decrease_factor, device=device)
                    if column == 'val_loss_hard_bitwise_minmax_mlp':
                        result_dict['val_loss_hard_bitwise_minmax_mlp'].append(val_loss_hard_bitwise_minmax_mlp)
                    elif column == 'val_loss_hard_bitwise_quantile_mlp':
                        result_dict['val_loss_hard_bitwise_quantile_mlp'].append(val_loss_hard_bitwise_quantile_mlp)        
            results_df = pd.DataFrame(result_dict) 
            results_df.to_csv(join(result_folder, f'{dataset}_best_models_{n_bits}bits_{f+1}runs_ext.csv'), index=False)
            if f > 0:
                os.remove(join(result_folder, f'{dataset}_best_models_{n_bits}bits_{f}runs_ext.csv'))
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Number of iterations')

    parser.add_argument('--scratch', type=str, required=True,
                        help='Number of iterations')
    parser.add_argument('--n_runs', type=int, default=2,
                        help='Number of runs per model')
    parser.add_argument('--n_steps', type=int, default=100,)
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
    n_runs = args.n_runs
    n_bits_min = args.n_bits_min
    n_bits_max = args.n_bits_max
    n_steps = args.n_steps
    result_folder = args.result_folder
    csv_folder = args.csv_folder
    if csv_folder is None:
        csv_folder = result_folder


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Start evaluating best models for {dataset} dataset")
    results_df = train_and_evaluate_best_models_multiple_bits(csv_folder=csv_folder, 
                                                            dataset=dataset, result_folder=result_folder,
                                                            n_runs=n_runs, 
                                                            n_bits_min=n_bits_min, n_bits_max=n_bits_max, 
                                                            device=device)    
    # results_df.to_csv(join(result_folder, f'{dataset}_best_models__{n_bits}bits_{n_runs}runs_Final.csv'), index=False)
    print(f"Finished evaluating best models for {dataset} dataset")

