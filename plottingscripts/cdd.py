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
    # plt.title('Critical difference diagram of average score ranks')
    avg_rank = results_df.groupby('dataset_name').accuracy.rank(pct=True).groupby(results_df.classifier_name).mean()
    sp.critical_difference_diagram(avg_rank, test_results)
    plt.savefig('plottingscripts/figures/' + str(name) +'.png')
    plt.show()

results_list_avg_accuracy = []
results_list_accuracy = []

datasets = ['california', 'fried', 'superconduct', 'wine_quality']
model_counters = {'Full precision': 0, 'Post minmaxQ': 0, 'Post quantileQ':0, 'Pre minmaxQ': 0, 'Pre quantileQ': 0,
              'SoftQ': 0, 'HardQ': 0, 'Bitwise softQ': 0, 'Bitwise hardQ': 0}

for data in datasets:
    df = pd.read_csv('results/' + f'{data}' + '_best_models__4bits_20runs.csv', sep=',', index_col=False)

    methods = {'Full precision': df.iloc[0],
              'Post minmaxQ': df.iloc[1],
              'Post quantileQ': df.iloc[2],
              'Pre minmaxQ': df.iloc[3],
              'Pre quantileQ': df.iloc[4],
              'SoftQ': df.iloc[5],
              'HardQ': df.iloc[5],
              'Bitwise softQ': df.iloc[6],
              'Bitwise hardQ': df.iloc[6]}

    for key in methods:
        column = methods[key]
        min_values = column.min()
        counter = 0
        meanvalue_acc = 0
        for value in column:
            meanvalue_acc += value
            counter += 1
        model_counters[key] += counter
        results_list_accuracy.append({'classifier_name': key, 'dataset_name': data, 'accuracy': min_values})
        results_list_avg_accuracy.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_acc/counter})

create_critical_difference_diagram(results_list_accuracy, 'Accuracy')
create_critical_difference_diagram(results_list_avg_accuracy, 'Average Accuracy')
