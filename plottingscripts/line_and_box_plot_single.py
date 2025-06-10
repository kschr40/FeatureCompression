import os
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import re
# In case you want to check which font are available at your system
# import matplotlib.font_manager
# fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation=45)
    ax.set_xlim(0.25, len(labels) + 0.75)

pd.set_option('display.max_columns', None)
def create_boxplot_bits(ax, results : pd.DataFrame, name : str, loss_columns : [], hline : int = 0, ymax : float = None):
    boxplot = results[loss_columns].boxplot(figsize=(12, 8), ax=ax)
    ax.axhline(y=hline, linestyle='--', color=(1, 0, 0, 0.5))
    if ymax is not None:
        boxplot.set_ylim([0, ymax])

def create_violinplot_bits(ax, results : pd.DataFrame, name : str, loss_columns : [], hline : int = 0, ymax : float = None):
    ax.violinplot(results[loss_columns],
                  showmeans=False,
                  showmedians=True)
    ax.axhline(y=hline, linestyle='--', color=(1, 0, 0, 0.5))
    if ymax is not None:
        ax.set_ylim([0, ymax])
    set_axis_style(ax, loss_columns)

datasets = ['wine', 'california'] # 'superconduct', 'wine_quality']
num_figures = len(datasets)
font = {'family' : 'Arial',
        'size'   : 22}

plt.rc('font', **font)
plt.rcParams['image.cmap'] = 'viridis'
counter = 0

if not os.path.exists('plottingscripts/figures/boxplot/'):
    os.makedirs('plottingscripts/figures/boxplot/')
if not os.path.exists('plottingscripts/figures/lineplot/'):
    os.makedirs('plottingscripts/figures/lineplot/')
if not os.path.exists('plottingscripts/figures/vioplot/'):
    os.makedirs('plottingscripts/figures/vioplot/')
loss_columns = ['Full precision', 'Post minmaxQ', 'Post quantileQ', 'Pre minmaxQ', 'Pre quantileQ', 'SoftQ',
                'HardQ', 'Bitwise softQ', 'Bitwise hardQ']

for data in datasets:
    path = f'results/processed_kFold_results/avg_by_hyper/' + f'{data}.csv'
    dataframe = pd.read_csv(path, sep=',', index_col=False, on_bad_lines='warn')
    prefered_hyperparameter = dataframe[dataframe['Full precision'] == dataframe['Full precision'].min()]
    number = prefered_hyperparameter['hyperparameter_setting_id'].values[0]
    dfsubset = dataframe[dataframe['hyperparameter_setting_id'] == number]
    if counter == 0:
        legend = False
    else:
        legend = True
    if data == 'california':
        ylim = (0.3, 0.9)
    elif data == 'wine':
        ylim = (0.4, 1.1)
    else:
        ylim = (0, 1.5)
    fig, ax = plt.subplots(1, 1, figsize=((12), 8), dpi=100)
    leftoverlines =  ['Post minmaxQ', 'Post quantileQ', 'Pre minmaxQ', 'Pre quantileQ']
    softq =  ['SoftQ', 'Bitwise softQ']
    hardq = ['HardQ', 'Bitwise hardQ']
    full_precision = ['Full precision']
    viridis = plt.colormaps['viridis']
    colors = [viridis(i) for i in np.linspace(0, 1, 9)]
    dfsubset.plot(x='bits', y=leftoverlines, kind='line', linestyle='--', marker='o', ax=ax, legend=legend, ylim=ylim, fontsize=22, color=colors[0:4])
    dfsubset.plot(x='bits', y=full_precision, color='r', kind='line', marker='o', ax=ax, legend=legend, ylim=ylim, fontsize=22)
    dfsubset.plot(x='bits', y=softq, kind='line', linestyle='-.', marker='o', ax=ax, legend=legend, ylim=ylim, fontsize=22, color=[colors[5], colors[7]])
    dfsubset.plot(x='bits', y=hardq, kind='line', linestyle=':', marker='o', ax=ax, legend=legend, ylim=ylim, fontsize=22, color=[colors[6], colors[8]])
    ax.set_xlabel('Bits')
    ax.set_ylabel('MSE Loss')
    ax.set_xticks(np.arange(2, 9, step=1))
    counter += 1
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(f'plottingscripts/figures/lineplot/{data}.png')
    plt.show()
    plt.close(fig)
counter = 0
for data in datasets:
    path = f'results/processed_kFold_results/avg_by_hyper/' + f'{data}.csv'
    dataframe = pd.read_csv(path, sep=',', index_col=False, on_bad_lines='warn')
    dfsubset4 = dataframe[dataframe['bits'] == 4]
    hline = dfsubset4['Full precision'].median()
    fig, ax = plt.subplots(1, 1, figsize=((12), 8), dpi=100)
    plt.ylabel('MSE Loss')
    if data == 'california':
        create_boxplot_bits(ax, dfsubset4, f'{data}_avg_boxplot_bits',
                                             loss_columns=loss_columns, hline=hline, ymax=2)
    else:
        create_boxplot_bits(ax, dfsubset4, f'{data}_avg_boxplot_bits',
                                             loss_columns=loss_columns, hline=hline)
    ax.set_ylabel(None)
    ax.set_ylabel('MSE Loss')
    counter += 1
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(loss_columns))],
                  labels=loss_columns, rotation=45)
    fig.tight_layout()
    fig.savefig(f'plottingscripts/figures/boxplot/{data}.png')
    plt.show()
    plt.close(fig)
counter = 0
for data in datasets:
    path = f'results/processed_kFold_results/avg_by_hyper/' + f'{data}.csv'
    dataframe = pd.read_csv(path, sep=',', index_col=False, on_bad_lines='warn')
    dfsubset4 = dataframe[dataframe['bits'] == 4]
    hline = dfsubset4['Full precision'].median()
    fig, ax = plt.subplots(1, 1, figsize=((12), 8), dpi=100)
    if data == 'california':
        create_violinplot_bits(ax, dfsubset4, f'{data}_avg_boxplot_bits', loss_columns=loss_columns,
                           hline=hline, ymax=2)
    else:
        create_violinplot_bits(ax, dfsubset4, f'{data}_avg_boxplot_bits', loss_columns=loss_columns, hline=hline)
    ax.set_ylabel('MSE Loss')
    counter += 1
    plt.tight_layout()
    fig.savefig(f'plottingscripts/figures/vioplot/{data}.png')
    plt.show()
    plt.close(fig)