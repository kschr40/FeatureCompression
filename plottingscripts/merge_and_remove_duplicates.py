import pandas as pd
import os
import re

datasets = ['california', 'cpu_act', 'fried', 'sulfur', 'superconduct', 'wine']
fulldataset = pd.DataFrame()
full_min_per_hyper = pd.DataFrame()
full_avg_per_hyper = pd.DataFrame()
lastname = ''
for data in datasets:
    path = f'results/raw_kFold_results/{data}/'
    bits = [2,3,4,5,6,7,8]
    for bit in bits:
        df = pd.DataFrame()
        for name in os.listdir(path):
            regex_bits = re.search(r'\dbits', name)
            regex_bits = regex_bits.group(0) if regex_bits else None
            if regex_bits is None:
                continue
            if regex_bits is None:
                ValueError(f"File {name} does not contain 'Xbits' in its name.")
            bits_number = regex_bits.replace('bits', '')
            bits = int(regex_bits[0])
            if bits == bit:
                tmp_df = pd.read_csv(path+name, sep=',', index_col=False, on_bad_lines='warn')
                df = pd.concat([tmp_df, df], ignore_index=True)
                lastname = name
                if not os.path.exists(f'{path}/archive'):
                    os.makedirs(f'{path}/archive')
                os.replace(f"{path}/{name}", f"{path}/archive/{name}")
        df.drop_duplicates(inplace=True)
        if not df.empty:
            df.sort_values(by=['hyperparameter_setting_id', 'weight_decay','learning_rate','hidden_layers','hidden_neurons','num_epochs','decrease_factor', 'kFold_id'], ascending=True, inplace=True)
            df.to_csv(f'{path}/{lastname}', sep=',', index=False, header=True)

