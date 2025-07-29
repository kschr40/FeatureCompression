import pandas as pd
import os
import re
import numpy as np
datasets = ['california', 'cpu_act', 'fried', 'sulfur', 'superconduct', 'wine']
fulldataset = pd.DataFrame()
full_min_per_hyper = pd.DataFrame()
full_avg_per_hyper = pd.DataFrame()

for data in datasets:
    path = 'results/raw_kFold_results/' + f'{data}/'
    best_average = pd.DataFrame()
    best_min = pd.DataFrame()
    datasetspecfic = pd.DataFrame()
    all_avg_by_hyper = pd.DataFrame()
    needed_columns = ['hyperparameter_setting_id', 'bits', 'weight_decay', 'learning_rate',
                      'hidden_layers', 'hidden_neurons', 'num_epochs', 'decrease_factor']
    bits_number = 0
    loss_columns = labels=['FP', 'Po-MQ', 'Po-QQ', 'Pr-MQ', 'Pr-QQ', 'Bw-MQ', 'Bw-QQ', 'SQ', 'Bw-SQ']
    keep_columns = ['dataset', 'bits', 'weight_decay', 'learning_rate', 'hidden_layers', 'hidden_neurons', 'num_epochs', 'decrease_factor']

    for name in os.listdir(path):
        regex_bits = re.search(r'\dbits', name)
        regex_bits = regex_bits.group(0) if regex_bits else None
        if isinstance(regex_bits, type(None)):
            print(f"File {name} does not contain 'Xbits' in its name.")
            continue
        bits_number = regex_bits.replace('bits', '')
        bits = int(regex_bits[0])
        df = pd.read_csv(path+name, sep=',', index_col=False, on_bad_lines='warn')
        repeated_numbers = []
        counter = 0
        for x in range(len(df) // 10):
            repeated_numbers.extend(np.repeat(counter, 10))
            counter += 1
        # Ensures that the hyperparameter id is not used twice for different bits.
        df['hyperparameter_setting_id'] = repeated_numbers
        df = df.rename(columns={'val_loss_mlp': 'FP', 'val_loss_hard_post_mlp':  'Po-MQ',
                                'val_loss_hard_thr_post_mlp': 'Po-QQ', 'val_loss_hard_pre_mlp': 'Pr-MQ',
                                'val_loss_hard_thr_pre_mlp': 'Pr-QQ', 'val_loss_soft_mlp':'SoftQ',
                                'val_loss_soft_hard_mlp':'SQ', 'val_loss_soft_comp_mlp': 'Bitwise softQ', 'val_loss_soft_hard_comp_mlp':'Bw-SQ',
                                'val_loss_hard_bitwise_quantile_mlp': 'Bw-QQ', 'val_loss_hard_bitwise_minmax_mlp': 'Bw-MQ'})
        df = df.rename(columns={'train_loss_mlp': 'Train full precision', 'train_loss_hard_post_mlp':  'Train post minmaxQ',
                                'train_loss_hard_thr_post_mlp': 'Train post quantileQ', 'train_loss_hard_pre_mlp': 'Train pre minmaxQ',
                                'train_loss_hard_thr_pre_mlp': 'Train pre quantileQ', 'train_loss_soft_mlp':'Train SoftQ',
                                'train_loss_soft_hard_mlp':'Train HardQ', 'train_loss_soft_comp_mlp': 'Train Bitwise softQ',
                                'train_loss_soft_hard_comp_mlp':'Train Bitwise hardQ', 'train_loss_hard_bitwise_quantile_mlp': 'Train Bitwise quantileQ',
                                'train_loss_hard_bitwise_minmax_mlp': 'Train Bitwise minmaxQ'})
        df['bits'] = [bits] * df.shape[0]
        df['dataset'] = [data] * df.shape[0]
        df = df.head(500)
        for col in loss_columns:
            avg_value = df[col].mean()
            df[col + '_avg'] = avg_value
        for col in loss_columns:
            avg_value = df[col].min()
            df[col + '_min'] = avg_value
        for col in loss_columns:
            df[col + '_avg_by_hyperparam'] = (
                df.groupby(['hyperparameter_setting_id', 'bits', 'weight_decay','learning_rate','hidden_layers','hidden_neurons','num_epochs','decrease_factor'])[col]
                .transform('mean')
            )
        for col in loss_columns:
            df[col + '_min_by_hyperparam'] = (
                df.groupby(['hyperparameter_setting_id', 'bits', 'weight_decay','learning_rate','hidden_layers','hidden_neurons','num_epochs','decrease_factor'])[col]
                .transform('min')
            )
        new_df = pd.DataFrame(df[needed_columns].iloc[:])

        averageandhyper = df.apply(lambda row:
                                   df[(df['hyperparameter_setting_id'] == row[
                                       'hyperparameter_setting_id']) &
                                      (df['weight_decay'] == row['weight_decay']) &
                                      (df['bits'] == row['bits']) &
                                      (df['learning_rate'] == row['learning_rate']) &
                                      (df['hidden_layers'] == row['hidden_layers']) &
                                      (df['hidden_neurons'] == row['hidden_neurons']) &
                                      (df['num_epochs'] == row['num_epochs']) &
                                      (df['decrease_factor'] == row['decrease_factor'])
                                      ][loss_columns].mean(), axis=1)

        new_df[loss_columns] = averageandhyper
        grouped_df = df.groupby(needed_columns)[loss_columns].mean().reset_index()
        all_avg_by_hyper = pd.concat([all_avg_by_hyper, grouped_df])

        datasetspecfic = pd.concat([datasetspecfic, df], ignore_index=True)
        average = df.groupby(['hyperparameter_setting_id', 'bits'])[loss_columns].mean().reset_index()
        average['dataset'] = [data] * average.shape[0]
        best_average = pd.concat([best_average, average], ignore_index=True)
        min = df.groupby(['hyperparameter_setting_id', 'bits'])[loss_columns].min().reset_index()
        min['dataset'] = [data] * min.shape[0]
        best_min = pd.concat([best_min, min], ignore_index=True)
    if not os.path.exists('results/processed_kFold_results/fulldata/'):
        os.makedirs('results/processed_kFold_results/fulldata/')
    if not os.path.exists('results/processed_kFold_results/min_by_hyper/'):
        os.makedirs('results/processed_kFold_results/min_by_hyper/')
    if not os.path.exists('results/processed_kFold_results/avg_by_hyper/'):
        os.makedirs('results/processed_kFold_results/avg_by_hyper/')

    fulldataset = pd.concat([fulldataset, datasetspecfic], ignore_index=True)
    datasetspecfic.to_csv(f'results/processed_kFold_results/fulldata/{data}_fulldata.csv', index=False)
    best_min = best_min.sort_values(by=['hyperparameter_setting_id', 'bits'])
    best_min.to_csv(f'results/processed_kFold_results/min_by_hyper/{data}.csv', index=False)
    best_average = best_average.sort_values(by=['hyperparameter_setting_id', 'bits'])
    best_average.to_csv(f'results/processed_kFold_results/avg_by_hyper/{data}.csv', index=False)
    all_avg_by_hyper = all_avg_by_hyper.sort_values(by=['hyperparameter_setting_id', 'bits'])
    all_avg_by_hyper.to_csv(f'results/processed_kFold_results/avg_by_hyper/{data}_hyperparameter.csv', index=False)
    full_min_per_hyper = pd.concat([full_min_per_hyper, best_min], ignore_index=True)
    full_avg_per_hyper = pd.concat([full_avg_per_hyper, best_average], ignore_index=True)


