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

def random_search_soft_quantization_threshold(train_loader, val_loader, 
                                              result_folder, dataset,  
                                              n_steps = 10, n_bits =8, num_features=8, optimize_dict = {}, device = 'cpu'):
    
    thresholds = get_quantization_thresholds(train_loader, n_bits)
    min_values, max_values = get_min_max_values(train_loader, num_features=num_features)
    
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
    

    hyperparameter_dict = {
        'weight_decay': [],
        'learning_rate': [],
        'hidden_layers': [],
        'hidden_neurons': [],
        'num_epochs': [],
        'decrease_factor': []}
    

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
            elif key == 'num_epochs':
                num_epochs = random.choice(value)    
            elif key == 'add_noise':
                add_noise = random.choice(value)    
            elif key == 'decrease_factor':
                decrease_factor = random.choice(value)
            else:
                raise ValueError(f"Unknown hyperparameter: {key}")
            
        architecture = [num_features] + [hidden_neurons] * hidden_layers + [1]
        hyperparameter_dict['weight_decay'].append(weight_decay)
        hyperparameter_dict['learning_rate'].append(learning_rate)
        hyperparameter_dict['hidden_layers'].append(hidden_layers)
        hyperparameter_dict['hidden_neurons'].append(hidden_neurons)
        hyperparameter_dict['num_epochs'].append(num_epochs)
        hyperparameter_dict['decrease_factor'].append(decrease_factor)

        # Calculate losses for mlp model
        val_loss_mlp, val_loss_hard_post_mlp, val_loss_hard_thr_post_mlp = train_mlp_model(train_loader=train_loader, val_loader=val_loader,
            architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, device=device)
        
        # Calculate losses for pre-training quantization model
        val_loss_hard_pre_mlp, val_loss_hard_thr_pre_mlp = train_mlp_pre_model(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, device=device)

        # Calculate losses for quantization model
        val_loss_soft_mlp, val_loss_soft_hard_mlp = train_soft_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, decrease_factor=decrease_factor, device=device)

        # Calculate losses for quantization model
        val_loss_soft_comp_mlp, val_loss_soft_hard_comp_mlp = train_soft_comp_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
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
 
        losses_df = pd.DataFrame({
            'val_loss_mlp': val_loss_mlp_values,
            'val_loss_hard_post_mlp': val_loss_hard_post_mlp_values,
            'val_loss_hard_thr_post_mlp': val_loss_hard_thr_post_mlp_values,
            'val_loss_hard_pre_mlp': val_loss_hard_pre_mlp_values,
            'val_loss_hard_thr_pre_mlp': val_loss_hard_thr_pre_mlp_values,
            'val_loss_soft_mlp': val_loss_soft_mlp_values,
            'val_loss_soft_hard_mlp': val_loss_soft_hard_mlp_values,
            'val_loss_soft_comp_mlp': val_loss_soft_comp_mlp_values,
            'val_loss_soft_hard_comp_mlp': val_loss_soft_hard_comp_mlp_values
        })

        # Create DataFrame with results
        results_df = pd.DataFrame(hyperparameter_dict)
        results_df = pd.concat([results_df, losses_df], axis=1)

        results_df = results_df.sort_values('val_loss_mlp')  # Sort by loss ascending  
        results_df.to_csv(f'{result_folder}/{dataset}_hyperparameter_tuning_{n_bits}bits_{f+1}steps.csv', index=False)
        # Delete the intermediate CSV file
        if f > 0:
            os.remove(f'{result_folder}/{dataset}_hyperparameter_tuning_{n_bits}bits_{f}steps.csv')

    return results_df

# def load_data(datasetname, scratch):
#     data_folder = scratch
#     X_file = os.path.join(data_folder, datasetname + "X.npy")
#     y_file = os.path.join(data_folder, datasetname + "Y.npy")
#     # Check if the dataset already exists
#     if os.path.exists(X_file) and os.path.exists(y_file):
#         X = np.load(X_file, allow_pickle=True)
#         y = np.load(y_file, allow_pickle=True)
#     else:
#         dataset = openml.datasets.get_dataset(dataset_id=datasetname, version=1)
#         X, y,categorical_indicator, attribute_names= dataset.get_data(target=dataset.default_target_attribute)
#         X = X.T[np.array(categorical_indicator) == False].T
#         # Ensure the data folder exists
#         os.makedirs(data_folder, exist_ok=True)
#         np.save(X_file, X)
#         np.save(y_file, y)
#     return process_data(X, y)

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
    args = parser.parse_args()
    dataset = args.dataset
    scratch = args.scratch
    n_steps = args.n_steps
    n_bits = args.n_bits
    result_folder = args.result_folder

    train_loader, val_loader, test_loader = load_data(dataset, scratch)
    for X, _ in train_loader:
        num_features = X.shape[1]
        break

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimize_dict = {'weight_decay': [0, 0.0001],
                    'learning_rate': [0.001, 0.0001],
                    'hidden_layers': [3,4,5,6],
                    'hidden_neurons': [128, 256, 512, 1024, 2048, 4096],
                    'num_epochs': [30,50,70],
                    'decrease_factor': [0.001, 0.0001]}
    
    print(f"Running random search for {n_steps} steps with {n_bits} bits on dataset {dataset}")
    results_df_all = random_search_soft_quantization_threshold( train_loader=train_loader,
                                                                val_loader=val_loader,
                                                                result_folder=result_folder,
                                                                dataset=dataset,
                                                                n_bits=n_bits,
                                                                n_steps=n_steps,
                                                                optimize_dict=optimize_dict,
                                                                device=device,
                                                                num_features=num_features)
    # results_df_all.to_csv(f'{result_folder}/{dataset}_hyperparameter_tuning_{n_bits}bits_{n_steps}steps_Final.csv', index=False)
    print(f"Finished random search for {n_steps} steps with {n_bits} bits on dataset {dataset}")