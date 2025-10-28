from datasets import get_min_max_values, get_quantization_thresholds, load_data, get_minmax_thresholds
import torch
import pandas as pd
import random
from tqdm import tqdm
from training import train_mlp_model, train_mlp_pre_model, train_soft_mlp, train_soft_comp_mlp, train_hard_comp_mlp, train_llt, train_lsq, train_soft_plus_mlp
import argparse
import os
import re
import glob
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

def hyperparameter_application(X_tensor : torch.tensor, y_tensor : torch.tensor, result_folder, dataset,
                               weight_decay = 0, learning_rate = 0.001, hidden_layers = 2, hidden_neurons = 256,
                               num_epochs = 30, decrease_factor = 0.001,
                                n_bits =8, device = 'cpu', debug=False, k_folds = 10, batch_size = 64):
    random_seeds = [42, 7, 21, 35, 14, 28, 56, 84, 63, 91]
    num_features = X_tensor.shape[1]

    results_df = pd.DataFrame(index = list(range(k_folds)))
    results_df['weight_decay'] = weight_decay
    results_df['learning_rate'] = learning_rate
    results_df['hidden_layers'] = hidden_layers
    results_df['hidden_neurons'] = hidden_neurons
    results_df['num_epochs'] = num_epochs
    results_df['decrease_factor'] = decrease_factor
    results_df['n_bits'] = n_bits

    # Perform random search
    # start = 0
    # filenames = glob.glob(f'{result_folder}/{dataset}_hyperparameter_application_{n_bits}bits_*')
    # for filename in filenames:
    #     storage_match = re.search(r'bits_(\d+)steps.csv', filename)
    #     if storage_match.group(1) is not None:
    #         if int(storage_match.group(1)) > start:
    #             start = int(storage_match.group(1))
    # if start > 0:
    #     previousdata = pd.read_csv(f'{result_folder}/{dataset}_hyperparameter_application_{n_bits}bits_{start}steps.csv')
    #     results_df = pd.concat([results_df, previousdata], axis=0)

    if debug:
        k_folds = 2
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in tqdm(enumerate(kfold.split(X_tensor)),total=k_folds):
        # Preprocess data

        # 1. Set a fixed random seed
        SEED = random_seeds[fold % len(random_seeds)]

        # 2. Set seeds for Python, NumPy, and PyTorch
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        thresholds = get_quantization_thresholds(train_loader, n_bits)
        min_values, max_values = get_min_max_values(train_loader, num_features=num_features)
        minmax_thresholds = get_minmax_thresholds(min_values, max_values, n_bits)

        architecture = [num_features] + [hidden_neurons] * hidden_layers + [1]

        # Calculate losses for mlp model
        val_loss_mlp, val_loss_hard_post_mlp, val_loss_hard_thr_post_mlp, train_loss_mlp, time_mlp = train_mlp_model(train_loader=train_loader, val_loader=val_loader,
            architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, device=device)

        # Calculate losses for pre-training quantization model
        val_loss_hard_pre_mlp, val_loss_hard_thr_pre_mlp, train_loss_hard_thr_pre_mlp_mm, train_loss_hard_thr_pre_mlp_q, time_prMQ = train_mlp_pre_model(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, device=device)

        # Calculate losses for soft quantization model
        val_loss_soft_mlp, val_loss_soft_hard_mlp, train_loss_soft, time_SQ = train_soft_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, decrease_factor=decrease_factor, device=device)
        
        # Calculate losses for soft quantization model + learnable quantized values
        val_loss_soft_plus_mlp, val_loss_soft_plus_hard_mlp, train_loss_plus_soft, time_SQ_plus = train_soft_plus_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, decrease_factor=decrease_factor, device=device)
        
        # # Calculate losses for LLT quantization model
        # val_loss_llt, val_loss_llt_training, loss_training_last_epoch, time_llt = train_llt(train_loader=train_loader, val_loader=val_loader,
        #                                                                           architecture=architecture, min_values=min_values, max_values=max_values,
        #                                                                           thresholds=thresholds,
        #                                                                           num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
        #                                                                           n_bits=n_bits, decrease_factor=decrease_factor, device=device)
        
        # Calculate losses for LSQ quantization model
        val_loss_lsq, loss_training_last_epoch, time_lsq = train_lsq(train_loader=train_loader, val_loader=val_loader,
                                                                                  architecture=architecture, min_values=min_values, max_values=max_values,
                                                                                  thresholds=thresholds,
                                                                                  num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                                                                                  n_bits=n_bits, decrease_factor=decrease_factor, device=device)
        
        # Calculate losses for bitwise soft quantization model
        val_loss_soft_comp_mlp, val_loss_soft_hard_comp_mlp, train_loss_soft_hard_comp, time_BwSQ = train_soft_comp_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, decrease_factor=decrease_factor, device=device)
        


        # Calculate losses for scalar, resp. bitwise, minmax, resp. quantile, quantization model
        val_loss_hard_bitwise_minmax_mlp, val_loss_hard_bitwise_quantile_mlp, train_loss_hard_bitwise_mm, train_loss_hard_bitwise_q, time_BwMQ = train_hard_comp_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, minmax_thresholds=minmax_thresholds, thresholds=thresholds,
                                                    num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                                                    n_bits=n_bits, decrease_factor=decrease_factor, device=device)



        results_df.loc[fold, 'val_loss_fp'] = val_loss_mlp
        results_df.loc[fold, 'val_loss_PoMQ'] = val_loss_hard_post_mlp
        results_df.loc[fold, 'val_loss_PoQQ'] = val_loss_hard_thr_post_mlp
        results_df.loc[fold, 'val_loss_PrMQ'] = val_loss_hard_pre_mlp
        results_df.loc[fold, 'val_loss_PrQQ'] = val_loss_hard_thr_pre_mlp
        results_df.loc[fold, 'val_loss_SQ_train'] = val_loss_soft_mlp
        results_df.loc[fold, 'val_loss_SQ_inf'] = val_loss_soft_hard_mlp
        results_df.loc[fold, 'val_loss_BwSQ_train'] = val_loss_soft_comp_mlp
        results_df.loc[fold, 'val_loss_BwSQ_inf'] = val_loss_soft_hard_comp_mlp
        results_df.loc[fold, 'val_loss_SQplus_train'] = val_loss_soft_plus_mlp
        results_df.loc[fold, 'val_loss_SQplus_inf'] = val_loss_soft_plus_hard_mlp
        results_df.loc[fold, 'val_loss_BwMQ'] = val_loss_hard_bitwise_minmax_mlp
        results_df.loc[fold, 'val_loss_BwQQ'] = val_loss_hard_bitwise_quantile_mlp
        # results_df.loc[fold, 'val_loss_LLT'] = val_loss_llt
        # results_df.loc[fold, 'val_loss_LLT_training'] = val_loss_llt_training
        results_df.loc[fold, 'val_loss_LSQ'] = val_loss_lsq

        results_df.loc[fold, 'time_mlp'] = time_mlp
        results_df.loc[fold, 'time_prMQ'] = time_prMQ
        results_df.loc[fold, 'time_SQ'] = time_SQ
        results_df.loc[fold, 'time_SQ_plus'] = time_SQ_plus
        results_df.loc[fold, 'time_BwSQ'] = time_BwSQ
        results_df.loc[fold, 'time_BwMQ'] = time_BwMQ
        # results_df.loc[fold, 'time_llt'] = time_llt
        results_df.loc[fold, 'time_lsq'] = time_lsq

        # results_df.drop_duplicates(inplace=True)
        folder = Path(f'{result_folder}')
        folder.mkdir(parents=True, exist_ok=True)
    # results_df.to_csv(f'{result_folder}/{dataset}_hyperparameter_tuning_{n_bits}bits_{f+1}steps.csv', index=False)
    # if f > 0:
    #     os.remove(f'{result_folder}/{dataset}_hyperparameter_tuning_{n_bits}bits_{f}steps.csv')
    # results_df = pd.DataFrame()
    results_df.to_csv(f'{result_folder}/{dataset}_hyperparameter_application_{n_bits}bits.csv', index=False)
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    # Add --dataset, --type, and --train argument
    parser.add_argument('--dataset', type=str, required=True,
                        help='Number of iterations')

    parser.add_argument('--scratch', type=str, required=True,
                        help='directory to save the results to')
    parser.add_argument('--n_bits_min', type=int, default=4,
                        help='Minimum number of bits for quantization')
    parser.add_argument('--n_bits_max', type=int, default=-1,
                        help='Maximum number of bits for quantization')
    parser.add_argument('--result_folder', type=str, default='results',
                        help='Folder to save results')
    parser.add_argument('--debug', action='store_true', help='Should be in debug modus?')
    parser.add_argument('--no-debug', action='store_false', dest='debug', help='Do not start in debug modus')
    parser.set_defaults(debug=False)
    # parser.add_argument('--weight_decay', type=float, default=0,
    #                     help='Weight decay for optimizer')
    # parser.add_argument('--learning_rate', type=float, default=0.0001,
    #                     help='Learning rate for optimizer')
    # parser.add_argument('--hidden_layers', type=int, default=8,
    #                     help='Number of hidden layers in the MLP')
    # parser.add_argument('--hidden_neurons', type=int, default=512,
    #                     help='Number of hidden neurons per layer in the MLP')
    # parser.add_argument('--num_epochs', type=int, default=50,
    #                     help='Number of epochs for training')
    # parser.add_argument('--decrease_factor', type=float, default=0.001,
    #                     help='Decrease factor for soft quantization')

    args = parser.parse_args()
    dataset = args.dataset
    scratch = args.scratch
    debug = args.debug
    result_folder = args.result_folder
    n_bits_min = args.n_bits_min
    n_bits_max = args.n_bits_max
    if n_bits_max == -1:
        n_bits_max = n_bits_min
    for n_bits in range(n_bits_min, n_bits_max + 1):    
        best_hyperparameters = pd.read_csv(f'results/best_hyperparameters.csv')
        best_setup = best_hyperparameters.query(f'dataset == "{dataset}" and method == "FP"')

        ind = best_setup['min'].argmin()

        weight_decay = best_setup.weight_decay.values[ind]
        learning_rate = best_setup.learning_rate.values[ind]
        hidden_layers = best_setup.hidden_layers.values[ind]
        hidden_neurons = best_setup.hidden_neurons.values[ind]
        num_epochs = best_setup.num_epochs.values[ind]
        if dataset == 'wine_quality':
            weight_decay = 0.01
            learning_rate = 0.0001
            hidden_layers = 3
            hidden_neurons = 512
            num_epochs = 70
        decrease_factor = 0.001
        batch_size = 512

        X_tensor, y_tensor = load_data(dataset, scratch, False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Running hyperparameter application with {n_bits} bits on dataset {dataset} on device {device}")
        print(f"Weights: decay={weight_decay}, lr={learning_rate}, hidden_layers={hidden_layers}, hidden_neurons={hidden_neurons}, "
            f"num_epochs={num_epochs}, decrease_factor={decrease_factor}")

        results_df_all = hyperparameter_application(X_tensor=X_tensor, y_tensor=y_tensor,
                                                        result_folder=result_folder,
                                                        dataset=dataset,
                                                        n_bits=n_bits,
                                                        weight_decay=weight_decay,
                                                        learning_rate=learning_rate,
                                                        hidden_layers=hidden_layers,
                                                        hidden_neurons=hidden_neurons,
                                                        num_epochs=num_epochs,
                                                        decrease_factor=decrease_factor,
                                                        device=device, debug=debug,
                                                        batch_size = batch_size)
        print(f"Finished hyperparameter application with {n_bits} bits on dataset {dataset}")
