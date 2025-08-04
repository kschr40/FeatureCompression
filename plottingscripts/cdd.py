import os
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp

pd.set_option('display.max_columns', None)
def create_critical_difference_diagram(results, name):
    results_df = pd.DataFrame(results)

    # https://github.com/hfawaz/cd-diagram/tree/master could also be an approach however it looks uglier and has different test.
    # other option https://github.com/mirkobunse/critdd this one can also have more lines!
    matrix = results_df.pivot(index='dataset_name', columns='classifier_name', values='accuracy')
    test_results = sp.posthoc_conover_friedman(matrix)
    plt.figure(figsize=(10, 2), dpi=100)
    avg_rank = results_df.groupby('dataset_name').accuracy.rank(pct=True).groupby(results_df.classifier_name).mean()
    sp.critical_difference_diagram(avg_rank, test_results)
    plt.tight_layout()
    plt.savefig('plottingscripts/figures/cdd/' + str(name) +'.pdf')

datasets = ['california', 'cpu_act', 'fried', 'sulfur', 'superconduct', 'wine']
model_counters = {'FP': 0, 'Po-MQ': 0, 'Po-QQ':0, 'Pr-MQ': 0, 'Pr-QQ': 0,
              'SQ': 0, 'Bw-MQ': 0, 'Bw-QQ': 0, 'Bw-SQ': 0}

bits = [2, 3, 4, 5, 6, 7, 8]

if not os.path.exists('plottingscripts/figures/cdd/'):
    os.makedirs('plottingscripts/figures/cdd/')
    if not os.path.exists('plottingscripts/figures/cdd/avg/'):
        os.makedirs('plottingscripts/figures/cdd/avg')
    if not os.path.exists('plottingscripts/figures/cdd/min/'):
        os.makedirs('plottingscripts/figures/cdd/min')

for bit in bits:
    results_list_avg_accuracy = []
    results_list_accuracy = []
    for data in datasets:
        path = f'results/processed_kFold_results/avg_by_hyper/' + f'{data}.csv'
        fulldata = pd.read_csv(path, sep=',', index_col=False, on_bad_lines='warn')
        df = fulldata[fulldata['bits'] == bit]
        print(df)
        methods = {
            'FP': df['FP'].tolist(),
            'Po-MQ': df['Po-MQ'].tolist(),
            'Po-QQ': df['Po-QQ'].tolist(),
            'Pr-MQ': df['Pr-MQ'].tolist(),
            'Pr-QQ': df['Pr-QQ'].tolist(),
            'SQ': df['SQ'].tolist(),
            'Bw-MQ': df['Bw-MQ'].tolist(),
            'Bw-QQ': df['Bw-QQ'].tolist(),
            'Bw-SQ': df['Bw-SQ'].tolist()
        }

        for key in methods:
            column = methods[key]
            min_values = min(column)
            counter = 0
            meanvalue_acc = 0
            for value in column:
                meanvalue_acc += value
                counter += 1
            model_counters[key] += counter
            results_list_accuracy.append({'classifier_name': key, 'dataset_name': data, 'accuracy': min_values})
            results_list_avg_accuracy.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_acc/counter})
    create_critical_difference_diagram(results_list_accuracy, f'min/{bit}_MSE')
    create_critical_difference_diagram(results_list_avg_accuracy, f'avg/{bit}_MSE')
