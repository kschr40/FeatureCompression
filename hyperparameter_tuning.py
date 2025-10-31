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

def random_search_cv(X_tensor : torch.tensor, y_tensor : torch.tensor, result_folder, dataset,
                                              k_folds = 4, n_steps = 10, n_bits =8, optimize_dict = {}, device = 'cpu', debug=False, onlyllt=False):
    if debug:
        k_folds = 2
    num_features = X_tensor.shape[1]

    X_cv_array, X_test_array, y_cv_array, y_test_array = train_test_split(X_tensor.numpy(), y_tensor.numpy(), test_size=0.1, random_state=42)


    hyperparameter_df = create_random_search_df(optimize_dict, k_folds=k_folds, n_steps=n_steps, n_bits=n_bits)
    methods = ['FP', 'Po-MQ', 'Po-QQ', 'Pr-MQ', 'Pr-QQ', 'SQ', 'SQplus', 'Bw-MQ', 'Bw-QQ', 'Bw-SQ', 'LLT', 'LSQ']
    val_losses = ['val_loss_' + method for method in methods]
    test_losses = ['test_loss_' + method for method in methods]
    train_losses = ['train_loss_' + method for method in methods if 'Po' not in method]
    times = ['time_' + method for method in methods if ('Po' not in method and 'QQ' not in method)]
    result_df = pd.DataFrame(columns = val_losses + test_losses + train_losses + times, index = range(n_steps * k_folds))
    result_df = pd.concat([hyperparameter_df, result_df], axis=1)
    # Define default hyperparameters
    # weight_decay =  0
    # learning_rate = 0.001
    # hidden_layers = 2
    # hidden_neurons = 256
    # num_epochs = 30
    # add_noise = False
    # decrease_factor = 0.001

    # Lists to store results
    val_loss_FP_values = []
    val_loss_PoMQ_values = []
    val_loss_PoQQ_values = []
    val_loss_hard_pre_mlp_values = []
    val_loss_hard_thr_pre_mlp_values = []
    val_loss_soft_mlp_values = []
    val_loss_soft_hard_mlp_values = []
    val_loss_soft_comp_mlp_values = []
    val_loss_soft_hard_comp_mlp_values = []
    val_loss_hard_bitwise_minmax_mlp_values = []
    val_loss_hard_bitwise_quantile_mlp_values = []

    train_loss_FP_values = []
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

    # results_df = pd.DataFrame()
    # Perform random search
    random.seed(datetime.now().timestamp())
    start = 0
    filenames = glob.glob(f'{result_folder}/{dataset}/{dataset}_hyperparameter_tuning_{n_bits}bits_*')
    for filename in filenames:
        storage_match = re.search(r'iterations_(\d+).csv', filename)
        if storage_match is not None:
            if storage_match.group(1) is not None:
                if int(storage_match.group(1)) > start:
                    start = int(storage_match.group(1))
    if start > 0:
        previousdata = pd.read_csv(f'{result_folder}/{dataset}/{dataset}_hyperparameter_tuning_{n_bits}bits_{start}iterations.csv')
        if len(previousdata) == start * k_folds:
            print(f"Resuming from previous run with {start} iterations.")
            result_df = previousdata
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    kfold_split = kfold.split(X_tensor)
    splits = list(kfold_split)
    for f, row in tqdm(result_df.iterrows(), total=result_df.shape[0]):
        if not row.isna().any():
            continue  # Skip already computed rows
        weight_decay = row['weight_decay']
        learning_rate = row['learning_rate']
        hidden_layers = row['hidden_layers']
        hidden_neurons = row['hidden_neurons']
        num_epochs = row['num_epochs']
        decrease_factor = row['decrease_factor']
        fold = row['kFold_id']
        
        SEED = 42 + fold + f * k_folds
        train_idx, val_idx = splits[fold]






    # for f in tqdm(range(start, n_steps)):
    #     for key, value in optimize_dict.items():
    #         if key == 'weight_decay':
    #             weight_decay = random.choice(value)
    #         elif key == 'learning_rate':
    #             learning_rate = random.choice(value)
    #         elif key == 'hidden_layers':
    #             hidden_layers = random.choice(value)
    #         elif key == 'hidden_neurons':
    #             hidden_neurons = random.choice(value)
    #             if debug:
    #                 hidden_neurons = 10
    #         elif key == 'num_epochs':
    #             num_epochs = random.choice(value)    
    #         elif key == 'add_noise':
    #             add_noise = random.choice(value)    
    #         elif key == 'decrease_factor':
    #             decrease_factor = random.choice(value)
    #         else:
    #             raise ValueError(f"Unknown hyperparameter: {key}")

        # k_folds = 10
        # if debug:
        #     k_folds = 2
        # counter = 0
        # for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tensor)):
            # Set a fixed random seed
        SEED = 42 + fold + f * k_folds
        # Set seeds for Python, NumPy, and PyTorch
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        # hyperparameter_setting_id.append(f)
        # kFold_id.append(counter)
        # counter += 1
        # Preprocess data
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
        X_test_array_scaled = scaler_X.transform(X_test_array)
        y_test_array_scaled = scaler_y.transform(y_test_array)

        X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_array, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_array, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_array_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_array_scaled, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


        quantile_thresholds = get_quantization_thresholds(train_loader, n_bits)
        min_values, max_values = get_min_max_values(train_loader, num_features=num_features)
        minmax_thresholds = get_minmax_thresholds(min_values, max_values, n_bits)

        architecture = [num_features] + [hidden_neurons] * hidden_layers + [1]
        hyperparameter_dict['weight_decay'].append(weight_decay)
        hyperparameter_dict['learning_rate'].append(learning_rate)
        hyperparameter_dict['hidden_layers'].append(hidden_layers)
        hyperparameter_dict['hidden_neurons'].append(hidden_neurons)
        hyperparameter_dict['num_epochs'].append(num_epochs)
        hyperparameter_dict['decrease_factor'].append(decrease_factor)

        # Calculate losses for FP and Post-quantization models
        if np.isnan(row['val_loss_FP']) and not onlyllt:
            val_loss_FP, val_loss_PoMQ, val_loss_PoQQ, test_loss_FP, test_loss_PoMQ, test_loss_PoQQ, train_loss_FP, time_FP = train_fp_model(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                architecture=architecture, min_values=min_values, max_values=max_values, quantile_thresholds=quantile_thresholds,
                num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                n_bits=n_bits, device=device)
            result_df.at[f, 'val_loss_FP'] = val_loss_FP
            result_df.at[f, 'val_loss_Po-MQ'] = val_loss_PoMQ
            result_df.at[f, 'val_loss_Po-QQ'] = val_loss_PoQQ
            result_df.at[f, 'test_loss_FP'] = test_loss_FP
            result_df.at[f, 'test_loss_Po-MQ'] = test_loss_PoMQ
            result_df.at[f, 'test_loss_Po-QQ'] = test_loss_PoQQ
            result_df.at[f, 'train_loss_FP'] = train_loss_FP
            result_df.at[f, 'time_FP'] = time_FP
        
        # Calculate losses for pre-training quantization model
        if np.isnan(row['val_loss_Pr-MQ']) and not onlyllt:
            val_loss_PrMQ, val_loss_PrQQ, test_loss_PrMQ, test_loss_PrQQ, train_loss_PrMQ, train_loss_PrQQ, time_PrMQ = train_pre_model(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, architecture=architecture,
                                                                                                        min_values=min_values, max_values=max_values, quantile_thresholds=quantile_thresholds,
                num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                n_bits=n_bits, device=device)
            result_df.at[f, 'val_loss_Pr-MQ'] = val_loss_PrMQ
            result_df.at[f, 'val_loss_Pr-QQ'] = val_loss_PrQQ
            result_df.at[f, 'test_loss_Pr-MQ'] = test_loss_PrMQ
            result_df.at[f, 'test_loss_Pr-QQ'] = test_loss_PrQQ
            result_df.at[f, 'train_loss_Pr-MQ'] = train_loss_PrMQ
            result_df.at[f, 'train_loss_Pr-QQ'] = train_loss_PrQQ
            result_df.at[f, 'time_Pr-MQ'] = time_PrMQ

        # Calculate losses for soft quantization model
        if np.isnan(row['val_loss_SQ']) and not onlyllt:
            val_loss_SQ_train, val_loss_SQ_inf, test_loss_SQ_train, test_loss_SQ_inf, train_loss_SQ, time_SQ = train_SQ(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, architecture=architecture, min_values=min_values, max_values=max_values, quantile_thresholds=quantile_thresholds,
                num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                n_bits=n_bits, decrease_factor=decrease_factor, device=device)
            result_df.at[f, 'val_loss_SQ'] = val_loss_SQ_inf
            result_df.at[f, 'val_loss_SQ_train'] = val_loss_SQ_train
            result_df.at[f, 'test_loss_SQ'] = test_loss_SQ_inf
            result_df.at[f, 'test_loss_SQ_train'] = test_loss_SQ_train
            result_df.at[f, 'train_loss_SQ'] = train_loss_SQ
            result_df.at[f, 'time_SQ'] = time_SQ
        
        # Calculate losses for soft quantization model
        if np.isnan(row['val_loss_SQplus']) and not onlyllt:
            val_loss_SQPlus_train, val_loss_SQPlus_inf, test_loss_SQPlus_train, test_loss_SQPlus_inf, train_loss_SQPlus, time_SQPlus = train_SQplus(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, architecture=architecture, min_values=min_values, max_values=max_values, quantile_thresholds=quantile_thresholds,
                num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                n_bits=n_bits, decrease_factor=decrease_factor, device=device)
            result_df.at[f, 'val_loss_SQplus'] = val_loss_SQPlus_inf
            result_df.at[f, 'val_loss_SQplus_train'] = val_loss_SQPlus_train
            result_df.at[f, 'test_loss_SQplus'] = test_loss_SQPlus_inf
            result_df.at[f, 'test_loss_SQplus_train'] = test_loss_SQPlus_train
            result_df.at[f, 'train_loss_SQplus'] = train_loss_SQPlus
            result_df.at[f, 'time_SQplus'] = time_SQPlus

        # Calculate losses for LSQ quantization model
        if np.isnan(row['val_loss_LSQ']) and not onlyllt:
            val_loss_LSQ, test_loss_LSQ,train_loss_LSQ, time_LSQ = train_lsq(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                                                  architecture=architecture, min_values=min_values, max_values=max_values,
                                                                                  quantile_thresholds=quantile_thresholds,
                                                                                  num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                                                                                  n_bits=n_bits, decrease_factor=decrease_factor, device=device)
            result_df.at[f, 'val_loss_LSQ'] = val_loss_LSQ
            result_df.at[f, 'test_loss_LSQ'] = test_loss_LSQ
            result_df.at[f, 'train_loss_LSQ'] = train_loss_LSQ
            result_df.at[f, 'time_LSQ'] = time_LSQ
            
        if np.isnan(row['val_loss_LLT']) and onlyllt:
            val_loss_LLT_train, val_loss_LLT_inf, test_loss_LLT_train, test_loss_LLT_inf, train_loss_LLT, time_LLT = train_llt(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                                                  architecture=architecture, min_values=min_values, max_values=max_values,
                                                                                  quantile_thresholds=quantile_thresholds,
                                                                                  num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                                                                                  n_bits=n_bits, decrease_factor=decrease_factor, device=device)
            result_df.at[f, 'val_loss_LLT'] = val_loss_LLT_inf
            result_df.at[f, 'val_loss_LLT_train'] = val_loss_LLT_train
            result_df.at[f, 'test_loss_LLT'] = test_loss_LLT_train
            result_df.at[f, 'test_loss_LLT_train'] = test_loss_LLT_inf
            result_df.at[f, 'train_loss_LLT'] = train_loss_LLT
            result_df.at[f, 'time_LLT'] = time_LLT    

        # Calculate losses for bitwise soft quantization model
        if np.isnan(row['val_loss_Bw-SQ']):
            val_loss_BwSQ_train, val_loss_BwSQ_inf, test_loss_BwSQ_train, test_loss_BwSQ_inf, train_loss_BwSQ, time_BwSQ = train_BwSQ(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, architecture=architecture, min_values=min_values, max_values=max_values, quantile_thresholds=quantile_thresholds,
                num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                n_bits=n_bits, decrease_factor=decrease_factor, device=device)
            result_df.at[f, 'val_loss_Bw-SQ'] = val_loss_BwSQ_inf
            result_df.at[f, 'val_loss_Bw-SQ_train'] = val_loss_BwSQ_train
            result_df.at[f, 'test_loss_Bw-SQ'] = test_loss_BwSQ_inf
            result_df.at[f, 'test_loss_Bw-SQ_train'] = test_loss_BwSQ_train
            result_df.at[f, 'train_loss_Bw-SQ'] = train_loss_BwSQ
            result_df.at[f, 'time_Bw-SQ'] = time_BwSQ

        # Calculate losses for scalar, resp. bitwise, minmax, resp. quantile, quantization model
        if np.isnan(row['val_loss_Bw-MQ']):
            val_loss_BwMQ, val_loss_BwQQ, test_loss_BwMQ, test_loss_BwQQ, train_loss_BwMQ, train_loss_BwQQ, time_BwMQ = train_BwMQ_BwQQ(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, architecture=architecture, minmax_thresholds=minmax_thresholds, quantile_thresholds=quantile_thresholds,
                                                    num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                                                    n_bits=n_bits, decrease_factor=decrease_factor, device=device)
            result_df.at[f, 'val_loss_Bw-MQ'] = val_loss_BwMQ
            result_df.at[f, 'val_loss_Bw-QQ'] = val_loss_BwQQ
            result_df.at[f, 'test_loss_Bw-MQ'] = test_loss_BwMQ
            result_df.at[f, 'test_loss_Bw-QQ'] = test_loss_BwQQ
            result_df.at[f, 'train_loss_Bw-MQ'] = train_loss_BwMQ
            result_df.at[f, 'train_loss_Bw-QQ'] = train_loss_BwQQ
            result_df.at[f, 'time_Bw-MQ'] = time_BwMQ
        
        # result_df.at[f, 'val_loss_FP'] = val_loss_FP
        # result_df.at[f, 'val_loss_Po-MQ'] = val_loss_PoMQ
        # result_df.at[f, 'val_loss_Po-QQ'] = val_loss_PoQQ
        # result_df.at[f, 'val_loss_Pr-MQ'] = val_loss_PrMQ
        # result_df.at[f, 'val_loss_Pr-QQ'] = val_loss_PrQQ
        # result_df.at[f, 'val_loss_SQ'] = val_loss_SQ_inf
        # result_df.at[f, 'val_loss_SQplus'] = val_loss_SQPlus_inf
        # result_df.at[f, 'val_loss_Bw-SQ'] = val_loss_BwSQ_inf
        # result_df.at[f, 'val_loss_Bw-MQ'] = val_loss_BwMQ
        # result_df.at[f, 'val_loss_Bw-QQ'] = val_loss_BwQQ

        # result_df.at[f, 'val_loss_SQplus_train'] = val_loss_SQPlus_train
        # result_df.at[f, 'val_loss_SQ_train'] = val_loss_SQ_train
        # result_df.at[f, 'val_loss_Bw-SQ_train'] = val_loss_BwSQ_train

        # result_df.at[f, 'train_loss_FP'] = train_loss_FP
        # result_df.at[f, 'train_loss_Pr-MQ'] = train_loss_PrMQ
        # result_df.at[f, 'train_loss_Pr-QQ'] = train_loss_PrQQ
        # result_df.at[f, 'train_loss_SQ'] = train_loss_SQ
        # result_df.at[f, 'train_loss_SQplus'] = train_loss_SQPlus
        # result_df.at[f, 'train_loss_Bw-SQ'] = train_loss_BwSQ
        # result_df.at[f, 'train_loss_Bw-MQ'] = train_loss_BwMQ
        # result_df.at[f, 'train_loss_Bw-QQ'] = train_loss_BwQQ

        # result_df.at[f, 'time_FP'] = time_FP
        # result_df.at[f, 'time_Pr-MQ'] = time_PrMQ
        # result_df.at[f, 'time_Pr-QQ'] = time_PrMQ
        # result_df.at[f, 'time_SQ'] = time_SQ
        # result_df.at[f, 'time_SQplus'] = time_SQPlus
        # result_df.at[f, 'time_Bw-SQ'] = time_BwSQ
        # result_df.at[f, 'time_Bw-MQ'] = time_BwMQ
        

        # val_loss_FP_values.append(val_loss_FP)
        # val_loss_PoMQ_values.append(val_loss_PoMQ)
        # val_loss_PoQQ_values.append(val_loss_PoQQ)

        
        # val_loss_hard_pre_mlp_values.append(val_loss_hard_pre_mlp)
        # val_loss_hard_thr_pre_mlp_values.append(val_loss_hard_thr_pre_mlp)
        # val_loss_soft_mlp_values.append(val_loss_soft_mlp)
        # val_loss_soft_hard_mlp_values.append(val_loss_soft_hard_mlp)
        # val_loss_soft_comp_mlp_values.append(val_loss_soft_comp_mlp)
        # val_loss_soft_hard_comp_mlp_values.append(val_loss_soft_hard_comp_mlp)
        # val_loss_hard_bitwise_minmax_mlp_values.append(val_loss_hard_bitwise_minmax_mlp)
        # val_loss_hard_bitwise_quantile_mlp_values.append(val_loss_hard_bitwise_quantile_mlp)
        


        # train_loss_FP_values.append(train_loss_FP)
        
        # train_loss_hard_pre_mlp_values.append(train_loss_hard_thr_pre_mlp_mm)
        # train_loss_hard_thr_pre_mlp_values.append(train_loss_hard_thr_pre_mlp_q)
        # train_loss_soft_mlp_values.append(train_loss_soft)
        # train_loss_soft_comp_mlp_values.append(train_loss_soft_hard_comp)
        # train_loss_hard_bitwise_minmax_mlp_values.append(train_loss_hard_bitwise_mm)
        # train_loss_hard_bitwise_quantile_mlp_values.append(train_loss_hard_bitwise_q)
        
        # losses_df = pd.DataFrame({
        #     'hyperparameter_setting_id' : hyperparameter_setting_id,
        #     'kFold_id': kFold_id,
        #     'val_loss_FP': val_loss_FP_values,
        #     'val_loss_Po-MQ': val_loss_PoMQ_values,
        #     'val_loss_Po-QQ': val_loss_PoQQ_values,
        #     'val_loss_hard_pre_mlp': val_loss_hard_pre_mlp_values,
        #     'val_loss_hard_thr_pre_mlp': val_loss_hard_thr_pre_mlp_values,
        #     'val_loss_soft_mlp': val_loss_soft_mlp_values,
        #     'val_loss_soft_hard_mlp': val_loss_soft_hard_mlp_values,
        #     'val_loss_soft_comp_mlp': val_loss_soft_comp_mlp_values,
        #     'val_loss_soft_hard_comp_mlp': val_loss_soft_hard_comp_mlp_values,
        #     'val_loss_hard_bitwise_minmax_mlp': val_loss_hard_bitwise_minmax_mlp_values,
        #     'val_loss_hard_bitwise_quantile_mlp': val_loss_hard_bitwise_quantile_mlp_values,
        #     'train_loss_FP': train_loss_FP_values,
        #     'train_loss_hard_pre_mlp': train_loss_hard_pre_mlp_values,
        #     'train_loss_hard_thr_pre_mlp': train_loss_hard_thr_pre_mlp_values,
        #     'train_loss_soft_mlp': train_loss_soft_mlp_values,
        #     'train_loss_soft_comp_mlp': train_loss_soft_comp_mlp_values,
        #     'train_loss_hard_bitwise_minmax_mlp': train_loss_hard_bitwise_minmax_mlp_values,
        #     'train_loss_hard_bitwise_quantile_mlp': train_loss_hard_bitwise_quantile_mlp_values
        # })
        

        
        # Create DataFrame with results
        # if results_df is None:
        #     results_df = pd.DataFrame(hyperparameter_dict)
        #     results_df = pd.concat([results_df, losses_df], axis=1)
        # else:
        #     tmp = pd.DataFrame(hyperparameter_dict)
        #     tmp = pd.concat([tmp, losses_df], axis=1)
        #     results_df = pd.concat([results_df, tmp], ignore_index=True, axis=0)
        # results_df.drop_duplicates(inplace=True)
        # results_df = results_df.sort_values(['hyperparameter_setting_id', 'val_loss_FP'])  # Sort by loss ascending
        folder = Path(f'{result_folder}/{dataset}')
        folder.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(f'{result_folder}/{dataset}/{dataset}_hyperparameter_tuning_{n_bits}bits_{f+1}iterations.csv', index=False)
        if f > 0:
            os.remove(f'{result_folder}/{dataset}/{dataset}_hyperparameter_tuning_{n_bits}bits_{f}iterations.csv')
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    # Add --dataset, --type, and --train argument
    parser.add_argument('--dataset', type=str, required=True,
                        help='Number of iterations')

    parser.add_argument('--scratch', type=str, required=True,
                        help='directory to save the results to')
    parser.add_argument('--n_steps', type=int, default=2,
                        help='Number of iterations')
    parser.add_argument('--n_bits', type=int, default=4,
                        help='Number of bits for quantization')
    parser.add_argument('--result_folder', type=str, default='results',
                        help='Folder to save results')
    parser.add_argument('--debug', action='store_true', help='Should be in debug modus?')
    parser.add_argument('--no-debug', action='store_false', dest='debug', help='Do not start in debug modus')

    parser.add_argument('--onlyllt', action='store_true', help='Should be in only LLT modus?')
    parser.add_argument('--no-onlyllt', action='store_false', dest='onlyllt', help='Do not start in only LLT modus')
    parser.set_defaults(debug=False)
    parser.set_defaults(onlyllt=False)


    args = parser.parse_args()
    dataset = args.dataset
    scratch = args.scratch
    n_steps = args.n_steps
    n_bits = args.n_bits
    debug = args.debug
    onlyllt = args.onlyllt
    result_folder = args.result_folder

    X_tensor, y_tensor = load_data(dataset, scratch, False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimize_dict = {'weight_decay': [0, 0.0001, 0.001, 0.01],
                    'learning_rate': [0.001, 0.0001],
                    'hidden_layers': [5,6,8,10],
                    # 'hidden_layers': [3,5,7],
                    # 'hidden_layers': [1,2],
                    'hidden_neurons': [32,64,128,256,512,1024, 2048, 4096, 8192],
                    # 'hidden_neurons': [32,64,128,256,512],
                    # 'hidden_neurons': [2,3,4],
                    # 'num_epochs': [10],
                    'num_epochs': [30,50,70],
                    'decrease_factor': [0.001, 0.0001]
                    } ## Search Space for Hyperparameter Tuning, see Appendix F, Table 3
    if debug:
        optimize_dict['hidden_neurons'] = [5]
    # print(f"Running random search for {n_steps} steps with {n_bits} bits on dataset {dataset} on device {device}")

    results_df_all = random_search_cv(X_tensor=X_tensor, y_tensor=y_tensor,
                                                            result_folder=result_folder,
                                                            dataset=dataset,
                                                            n_bits=n_bits,
                                                            n_steps=n_steps,
                                                            optimize_dict=optimize_dict,
                                                            device=device, debug=debug, onlyllt=onlyllt)
    # print(f"Finished random search for {n_steps} steps with {n_bits} bits on dataset {dataset}")
