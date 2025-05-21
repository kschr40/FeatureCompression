import os
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


plt.show()

pd.set_option('display.max_columns', None)
def create_barplot_bits(results, name):
    x = np.arange(7)
    bits = ("2", "3", "4", "5", "6", "7", "8")
    width = 0.10
    multiplier = 0
    method_results = {
        'Full precision': results['Full precision'].tolist(),
        'Post minmaxQ': results['Post minmaxQ'].tolist(),
        'Post quantileQ': results['Post quantileQ'].tolist(),
        'Pre minmaxQ': results['Pre minmaxQ'].tolist(),
        'Pre quantileQ': results['Pre quantileQ'].tolist(),
        'SoftQ': results['SoftQ'].tolist(),
        'HardQ': results['HardQ'].tolist(),
        'Bitwise softQ': results['Bitwise softQ'].tolist(),
        'Bitwise hardQ': results['Bitwise hardQ'].tolist()
    }
    fig, ax = plt.subplots(layout='constrained')
    baseline = 0
    static_baseline = 0
    for attribute, measurement in method_results.items():
        if attribute == 'Full precision':
            static_baseline = sum(measurement)/7
        offset = width * multiplier
        rects = ax.bar(x + offset, [a_i - static_baseline for a_i in (measurement)], width, label=attribute)
        plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: round(x + static_baseline,2)))
        # Labels would be overlapping
        # ax.bar_label(rects, padding=3, fmt='%.2f')
        multiplier += 1

    ax.set_xticks(x + width, bits)
    ax.legend(loc='upper left', ncols=3)
    #results.plot(x='bits', y=loss_columns, kind='line', marker='o', figsize=(12, 8))
    plt.ylabel('MSE Loss')
    plt.savefig('plottingscripts/figures/barplots/' + str(name) +'.png')

    plt.show()

def create_lineplot_bits(results, name):
    loss_columns = ['Full precision', 'Post minmaxQ', 'Post quantileQ', 'Pre minmaxQ', 'Pre quantileQ', 'SoftQ', 'HardQ', 'Bitwise softQ', 'Bitwise hardQ']
    x = np.arange(7)
    bits = ("2", "3", "4", "5", "6", "7", "8")

    results.plot(x='bits', y=loss_columns, kind='line', marker='o', figsize=(12, 8))
    plt.ylabel('MSE Loss')
    plt.savefig('plottingscripts/figures/lineplots/' + str(name) +'.png')

    plt.show()
datasets = ['california', 'fried', 'superconduct', 'wine_quality']

for data in datasets:
    path = 'results/' + f'{data}' + '_evaluation_from_2_to_8bits_new.csv'
    df = pd.read_csv(path, sep=',', index_col=False)
    df = df.rename(columns={'val_loss_mlp': 'Full precision', 'val_loss_hard_post_mlp':  'Post minmaxQ',
                            'val_loss_hard_thr_post_mlp': 'Post quantileQ', 'val_loss_hard_pre_mlp': 'Pre minmaxQ',
                            'val_loss_hard_thr_pre_mlp': 'Pre quantileQ', 'val_loss_soft_mlp':'SoftQ',
                            'val_loss_soft_hard_mlp':'HardQ', 'val_loss_soft_comp_mlp': 'Bitwise softQ', 'val_loss_soft_hard_comp_mlp':'Bitwise hardQ'})
    create_barplot_bits(df, data)
    create_lineplot_bits(df, data)


