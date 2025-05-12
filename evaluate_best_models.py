from datasets import load_data, get_min_max_values, get_quantization_thresholds
import torch
import pandas as pd
import random
from tqdm import tqdm
from training import train_mlp_model, train_mlp_pre_model, train_soft_mlp, train_soft_comp_mlp
import argparse
from os.path import join

def train_and_evaluate_best_models(df, loss_columns,
                                   train_loader, test_loader,
                                   num_features,
                                   n_runs = 5, n_bits = 4, device = 'cuda'):
    
    min_values, max_values = get_min_max_values(train_loader, num_features)
    thresholds = get_quantization_thresholds(train_loader, n_bits)

    result_dict = {}
    for col in loss_columns:
        result_dict[col] = []
    for column in tqdm(loss_columns):
        df.sort_values(by=column, ascending=True, inplace=True)
        best_weight_decay = df['weight_decay'].values[0]
        best_learning_rate = df['learning_rate'].values[0]
        best_hidden_layers = df['hidden_layers'].values[0]
        best_hidden_neurons = df['hidden_neurons'].values[0]
        best_num_epochs = df['num_epochs'].values[0]
        best_decrease_factor = df['decrease_factor'].values[0]

        for f in tqdm(range(n_runs)):
            current_val_loss = 0
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
                    current_val_loss = val_loss_mlp
                elif column == 'val_loss_hard_post_mlp':
                    current_val_loss = val_loss_hard_post_mlp
                elif column == 'val_loss_hard_thr_post_mlp':
                    current_val_loss = val_loss_hard_thr_post_mlp
            elif column in ['val_loss_hard_pre_mlp',
                            'val_loss_hard_thr_pre_mlp']:
                architecture = [num_features] + [best_hidden_neurons]*best_hidden_layers + [1]
                val_loss_hard_pre_mlp, val_loss_hard_thr_pre_mlp = train_mlp_pre_model(train_loader=train_loader, val_loader=test_loader,
                            architecture=architecture,
                            min_values=min_values, max_values=max_values, thresholds=thresholds,
                            num_epochs=int(best_num_epochs), learning_rate=best_learning_rate,
                            weight_decay=best_weight_decay, add_noise=False, n_bits=n_bits, device=device)
                if column == 'val_loss_hard_pre_mlp':
                    current_val_loss = val_loss_hard_pre_mlp
                elif column == 'val_loss_hard_thr_pre_mlp':
                    current_val_loss = val_loss_hard_thr_pre_mlp
            elif column in ['val_loss_soft_mlp',
                            'val_loss_soft_hard_mlp']:
                architecture = [num_features] + [best_hidden_neurons]*best_hidden_layers + [1]
                val_loss_soft_mlp, val_loss_soft_hard_mlp = train_soft_mlp(train_loader=train_loader, val_loader=test_loader,
                            architecture=architecture,
                            min_values=min_values, max_values=max_values, thresholds=thresholds,
                            num_epochs=int(best_num_epochs), learning_rate=best_learning_rate,
                            weight_decay=best_weight_decay, add_noise=False, n_bits=n_bits,
                            decrease_factor=best_decrease_factor, device=device)
                if column == 'val_loss_soft_mlp':
                    current_val_loss = val_loss_soft_mlp
                elif column == 'val_loss_soft_hard_mlp':
                    current_val_loss = val_loss_soft_hard_mlp
            elif column in ['val_loss_soft_comp_mlp',
                            'val_loss_soft_hard_comp_mlp']:
                n_thresholds_per_feature = 2 ** n_bits - 1
                architecture = [num_features * n_thresholds_per_feature] + [best_hidden_neurons]*best_hidden_layers + [1]
                val_loss_soft_comp_mlp, val_loss_soft_hard_comp_mlp = train_soft_comp_mlp(train_loader=train_loader, val_loader=test_loader,
                            architecture=architecture,
                            min_values=min_values, max_values=max_values, thresholds=thresholds,
                            num_epochs=int(best_num_epochs), learning_rate=best_learning_rate,
                            weight_decay=best_weight_decay, add_noise=False, n_bits=n_bits,
                            decrease_factor=best_decrease_factor, device=device)
                if column == 'val_loss_soft_comp_mlp':
                    current_val_loss = val_loss_soft_comp_mlp
                elif column == 'val_loss_soft_hard_comp_mlp':
                    current_val_loss = val_loss_soft_hard_comp_mlp
            result_dict[column].append(current_val_loss)   
    results_df = pd.DataFrame(result_dict) 
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Number of iterations')

    parser.add_argument('--scratch', type=str, required=True,
                        help='Number of iterations')
    parser.add_argument('--n_runs', type=int, default=2,
                        help='Number of runs per model')
    parser.add_argument('--n_bits', type=int, default=4,
                        help='Number of bits for quantization')
    parser.add_argument('--n_steps', type=int, default=100,)
    parser.add_argument('--result_folder', type=str, default='results',
                        help='Folder to save results')
    parser.add_argument('--csv_folder', type=str, 
                        default=None, required=False)
    args = parser.parse_args()
    dataset = args.dataset
    scratch = args.scratch
    n_runs = args.n_runs
    n_bits = args.n_bits
    n_steps = args.n_steps
    result_folder = args.result_folder
    csv_folder = args.csv_folder
    if csv_folder is None:
        csv_folder = result_folder
    csv_path = join(csv_folder, f'{dataset}_hyperparameter_tuning_{n_bits}bits_{n_steps}steps.csv')

    train_loader, val_loader, test_loader = load_data(dataset, scratch)
    for X, _ in train_loader:
        num_features = X.shape[1]
        break

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    df = pd.read_csv(csv_path)
    loss_columns = [col for col in df.columns if 'loss' in col]

    print(f"Start evaluating best models for {dataset} dataset")
    results_df = train_and_evaluate_best_models(df, loss_columns, 
                                                train_loader=train_loader, test_loader=test_loader,
                                                num_features=num_features,
                                                n_runs=n_runs, n_bits=n_bits, device=device)    
    results_df.to_csv(join(result_folder, f'{dataset}_best_models__{n_bits}bits_{n_runs}runs.csv'), index=False)
    print(f"Finished evaluating best models for {dataset} dataset")

