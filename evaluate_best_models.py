from datasets import get_dataloader, get_min_max_values, get_quantization_thresholds
import torch
import pandas as pd
import random
from tqdm import tqdm
from training import train_mlp_model, train_mlp_pre_model, train_soft_mlp, train_soft_comp_mlp

def train_and_evaluate_best_models(df, loss_columns,
                                   n_steps = 5, n_bits = 4, device = 'cuda'):
    
    train_loader, val_loader, test_loader = get_dataloader(dataset = dataset)
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

        for f in tqdm(range(n_steps)):
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
    dataset = 'California_Housing'

    device=  'cuda'
    n_steps = 2
    n_bits = 4
    num_features = 8
    n_thresholds_per_feature = 2 ** n_bits - 1
    n_thresholds = n_thresholds_per_feature * num_features
    path = f'results/California_Housing/random_search_results_all_{n_bits}bits.csv'
    df = pd.read_csv(path)
    loss_columns = [col for col in df.columns if 'loss' in col]

    results_df = train_and_evaluate_best_models(df, loss_columns, 
                                                n_steps = n_steps, n_bits = n_bits, device = 'cuda')    
    results_df.to_csv(f'results/{dataset}/best_models_{n_steps}steps_{n_bits}bits.csv', index=False)

