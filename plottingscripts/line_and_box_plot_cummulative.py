import os
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick
import re
import argparse
import seaborn as sns
import matplotlib.font_manager
from IPython.core.display import HTML

def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

# In case you want to check which font are available at your system
# import matplotlib.font_manager
# fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

pd.set_option('display.max_columns', None)
def create_boxplot_bits(ax, results : pd.DataFrame, name : str, loss_columns : [], hline : int = 0, ymax : float = None, ymin : float = 0):
    ax.boxplot(results[loss_columns], ylim=[ymin, ymax])
    ax.axhline(y=hline, linestyle='--', color=(1, 0, 0, 0.5))
    #if ymax is not None:
        #boxplot.set_ylim([ymin, ymax])
    ax.set_title(name)

def initialize_figures(num_figures: int, layout: str):
    if layout == 'singlerow':
        fig, axes = plt.subplots(1, num_figures, figsize=(7, 7/1.5), dpi=1000)
    elif layout == 'squared':
        half = num_figures//2
        fig, axes = plt.subplots(2, half, figsize=(7, ((7/1.5))), dpi=1000)
    else:
        fig, axes = plt.subplots(1, num_figures, figsize=((7 * num_figures), 7/1.5), dpi=1000)
    return fig, axes

def make_lineplot(layout: str):
    counter = 0
    fig, axes = initialize_figures(num_figures_box, layout)
    for data in datasets:
        if layout == 'squared':
            currentaxes = axes[counter % 2, counter // 2]
        elif layout == 'singlerow':
            currentaxes = axes[counter]
        else:
            currentaxes = axes[counter]
        path = f'results/processed_kFold_results/avg_by_hyper/' + f'{data}.csv'
        dataframe = pd.read_csv(path, sep=',', index_col=False, on_bad_lines='warn')
        bits = [2, 3, 4, 5, 6, 7, 8]
        methods = ['Post minmaxQ', 'Post quantileQ', 'Pre minmaxQ', 'Pre quantileQ', 'HardQ', 'Bitwise hardQ', 'Bitwise quantileQ', 'Bitwise minmaxQ']
        values_per_bit = pd.DataFrame()
        for bit in bits:
            subset = dataframe[dataframe['bits'] == bit]
            prefered_hyperparameter = subset[subset['Full precision'] == subset['Full precision'].min()]
            for method in methods:
                prefered_hyperparameter[method] = subset[method].min()
            values_per_bit = pd.concat([values_per_bit, prefered_hyperparameter], ignore_index=True)
        if layout == 'squared' and counter == 3:
            legend = True
        elif layout == 'singlerow' and counter == len(datasets) - 1:
            legend = True
        else:
            legend = False
        minimum = values_per_bit['Full precision'].mean()
        if data == 'california':
            realminimum = minimum - ((0.9 - minimum) * 0.15)
            ylim = (realminimum, 0.9)
        elif data == 'wine':
            realminimum = minimum - ((1.1 - minimum) * 0.15)
            ylim = (realminimum, 1.1)
        elif data == 'fried':
            realminimum = minimum - ((0.1 - minimum) * 0.15)
            ylim = (realminimum, 0.1)
        elif data == 'superconduct':
            realminimum = minimum - ((0.3 - minimum) * 0.15)
            ylim = (realminimum, 0.3)
        elif data == 'sulfur':
            realminimum = minimum - ((1 - minimum) * 0.15)
            ylim = (realminimum, 1)
        elif data == 'cpu_act':
            realminimum = minimum - ((0.4 - minimum) * 0.15)
            ylim = (realminimum, 0.4)
        else:
            ylim = (0, 5)
        leftoverlines =  ['Post minmaxQ', 'Post quantileQ', 'Pre minmaxQ', 'Pre quantileQ']
        softq =  ['Bitwise minmaxQ', 'Bitwise quantileQ']
        hardq = ['HardQ', 'Bitwise hardQ']
        full_precision = ['Full precision']
        viridis = plt.colormaps['viridis']
        colors = [viridis(i) for i in np.linspace(0, 1, 9)]
        values_per_bit.plot(x='bits', y=leftoverlines, kind='line', linestyle='--', marker='o', ax=currentaxes, legend=legend, ylim=ylim, color=colors[0:4], ms=2, lw=0.5)
        #values_per_bit.plot(x='bits', y=full_precision, color='r', kind='line', marker='o', ax=currentaxes, legend=legend, ylim=ylim, fontsize=8)
        values_per_bit.plot(x='bits', y=softq, kind='line', linestyle='-.', marker='o', ax=currentaxes, legend=legend, ylim=ylim, color=[colors[5], colors[7]], ms=2, lw=0.5)
        values_per_bit.plot(x='bits', y=hardq, kind='line', linestyle=':', marker='o', ax=currentaxes, legend=legend, ylim=ylim, color=[colors[6], colors[8]],ms=2, lw=0.5)
        currentaxes.set_xlabel('Bits')
        currentaxes.axhline(y=values_per_bit['Full precision'].mean(), linestyle='--', color='r', label='Mean Full Precision', lw=0.5)
        if (counter % 2) == 0 and counter // 2 == 0 or (counter % 2) == 1 and counter // 2 == 0:
            currentaxes.set_ylabel('MSE Loss')
        currentaxes.set_xticks(np.arange(2, 9, step=1))
        currentaxes.set_title(data)
        counter += 1
        if legend:
            #plt.legend.append('Mean Full Precision')
            plt.legend(ncol=4,
                loc='lower center',
                bbox_to_anchor=(0, -1.45),
                fontsize=9)
            plt.tight_layout(rect=[0, 0, 1, 1])

    fig.savefig(f'plottingscripts/figures/lineplot/cum/lineplot_{layout}.png')
    #plt.show()
    plt.close(fig)

def make_boxplot(layout: str):
    counter = 0
    fig, axes = initialize_figures(num_figures_box, layout)
    for data in boxdatasets:
        if layout == 'squared':
            ax = axes[counter % 2, counter // 2]
        elif layout == 'singlerow':
            ax = axes[counter]
        else:
            ax = axes[counter]
        path = f'results/processed_kFold_results/avg_by_hyper/' + f'{data}.csv'
        dataframe = pd.read_csv(path, sep=',', index_col=False, on_bad_lines='warn')
        bit = 4
        dfsubset4 = dataframe[dataframe['bits'] == bit]
        hline = dfsubset4['Full precision'].median()
        subdata = dfsubset4[loss_columns]
        boxplotlist = []
        for loss_colum in loss_columns:
            name_list = subdata[loss_colum].tolist()
            boxplotlist.append(name_list)
        ax.boxplot(boxplotlist, positions=[0,1,2,3,4,5,6,7,8], labels=loss_columns)
        sns.swarmplot(data=boxplotlist, ax=ax, color='black', alpha=0.5, linewidth=0.5, size=3, orient='v', dodge=True)

        ymin = None
        if data == 'california':
            ymax = 2
            ymin = 0.25
        elif data == 'wine':
            ymax = 1
            ymin = 0.4
        elif data == 'fried':
            ymax = 0.15
            ymin = 0.025
        elif data == 'superconduct':
            ymax = 0.3
        elif data == 'sulfur':
            ymax = 0.8
            ymin = 0.01
        elif data == 'cpu_act':
            ymax = 0.2
        else:
            ymax = None
        ax.axhline(y=hline, linestyle='--', color=(1, 0, 0, 0.5))
        ax.set_ylim([ymin, ymax])
        ax.set_title(f'{data}')
        ax.set_ylabel(None)
        if (counter % 2) == 0 and counter // 2 == 0 or (counter % 2) == 1 and counter // 2 == 0:
            ax.set_ylabel('MSE Loss')
        counter += 1
        ax.yaxis.grid(True)
        ax.set_xticks([y for y in range(len(loss_columns))],
                      labels=loss_columns, rotation=45)
    fig.tight_layout()
    fig.savefig(f'plottingscripts/figures/boxplot/cum/{layout}_boxplot.png')
    # plt.show()
    plt.close(fig)

def make_boxplotbest(layout:str):
    counter = 0
    fig, axes = initialize_figures(num_figures_box, layout)
    for data in boxdatasets:
        if layout == 'squared':
            ax = axes[counter % 2, counter // 2]
        elif layout == 'singlerow':
            ax = axes[counter]
        else:
            ax = axes[counter]
        fullpath = f'results/processed_kFold_results/fulldata/{data}_fulldata.csv'
        fulldataframe = pd.read_csv(fullpath, sep=',', index_col=False, on_bad_lines='warn')
        avgpath = f'results/processed_kFold_results/avg_by_hyper/' + f'{data}.csv'
        avgdataframe = pd.read_csv(avgpath, sep=',', index_col=False, on_bad_lines='warn')
        bits = [4]
        methods = ['Full precision', 'Post minmaxQ', 'Post quantileQ', 'Pre minmaxQ', 'Pre quantileQ', 'HardQ',
                   'Bitwise hardQ', 'Bitwise quantileQ', 'Bitwise minmaxQ']
        values_per_bit = pd.DataFrame()
        hyperparameters = {}
        # Get the best hyperparameter setting for each method
        for bit in bits:
            subset = avgdataframe[avgdataframe['bits'] == bit]
            for method in methods:
                item = subset[subset[method] == subset[method].min()]
                hyperparameters[method] = item['hyperparameter_setting_id'].values[0]
        bit = 4
        subset = fulldataframe[fulldataframe['bits'] == bit]
        prefered_hyperparameter = pd.DataFrame()
        for method in methods:
            if prefered_hyperparameter.empty:
                prefered_hyperparameter = subset[subset['hyperparameter_setting_id'] == hyperparameters[method]]
                continue
            dfbest = subset[subset['hyperparameter_setting_id'] == hyperparameters[method]][method]
            prefered_hyperparameter[method] = dfbest.values
        values_per_bit = pd.concat([values_per_bit, prefered_hyperparameter], ignore_index=True)


        hline = values_per_bit['Full precision'].median()
        ax.axhline(y=hline, linestyle='--', color=(1, 0, 0, 0.5))

        if counter == len(datasets) - 1:
            legend = True
        else:
            legend = False
        if data == 'california':
            ymin = 0.3
            ymax = 1
        elif data == 'wine':
            ymin = 0.4
            ymax = 0.8
        elif data == 'fried':
            ymin = 0.04
            ymax = 0.06
        elif data == 'superconduct':
            ymin = 0.06
            ymax = 0.15
        elif data == 'sulfur':
            ymin = 0.05
            ymax = 0.6
        elif data == 'cpu_act':
            ymin = 0.015
            ymax = 0.06
        else:
            ymin = 0
            ymax = 5
        boxplotlist = []
        for loss_colum in loss_columns:
            name_list = values_per_bit[loss_colum].tolist()
            boxplotlist.append(name_list)

        len_col = len(loss_columns)
        ax.set_title(f'{data}')
        ax.boxplot(boxplotlist, positions=[0, 1, 2, 3, 4, 5, 6, 7, 8], labels=loss_columns)
        sns.swarmplot(data=boxplotlist, ax=ax, color='black', alpha=0.5, linewidth=0.2, size=1, orient='v', dodge=True)
        ax.set_ylabel(None)
        ax.set_ylim([ymin, ymax])
        if (counter % 2) == 0 and counter // 2 == 0 or (counter % 2) == 1 and counter // 2 == 0:
            ax.set_ylabel('MSE Loss')
        counter += 1
        ax.yaxis.grid(True)
        ax.set_xticks([y for y in range(len(loss_columns))],
                      labels=[1,2,3,4,5,6,7,8,9])
        fig.tight_layout()
    fig.savefig(f'plottingscripts/figures/boxplot/cum/{bit}_boxplot_kfolds_{layout}.png')
    # plt.show()
    plt.close(fig)

if __name__ == "__main__":

    print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    # Add --dataset, --type, and --train argument
    parser.add_argument('--lineplot', action='store_true', help='Create lineplot')
    parser.add_argument('--no-lineplot', action='store_false', dest='lineplot', help='Not create lineplot')
    parser.add_argument('--boxplot', action='store_true', help='Create boxplot')
    parser.add_argument('--no-boxplot', action='store_false', dest='boxplot', help='Not create boxplot')
    parser.add_argument('--boxplotbestfold', action='store_true', help='Create boxplotbestfold')
    parser.add_argument('--no-boxplotbestfold', action='store_false', dest='boxplotbestfold', help='Not create boxplotbestfold')
    parser.set_defaults(lineplot=True)
    parser.set_defaults(boxplot=False)
    parser.set_defaults(boxplotbestfold=True)

    args = parser.parse_args()

    lineplot = args.lineplot
    boxplot = args.boxplot
    boxplotbestfold = args.boxplotbestfold

    datasets = ['california', 'cpu_act', 'fried', 'sulfur', 'superconduct', 'wine'] # "NewFuel
    boxdatasets = ['california', 'cpu_act', 'fried', 'sulfur', 'superconduct', 'wine'] # "NewFuelCar" "superconduct"]
    num_figures = len(datasets)
    num_figures_box = len(boxdatasets)

    font = {'family' : 'Times New Roman',
            'size'   : 8}

    plt.rc('font', **font)
    plt.rcParams['image.cmap'] = 'viridis'
    if boxplot or boxplotbestfold:
        if not os.path.exists('plottingscripts/figures/boxplot/cum/'):
            os.makedirs('plottingscripts/figures/boxplot/cum/')
    if lineplot:
        if not os.path.exists('plottingscripts/figures/lineplot/cum/'):
            os.makedirs('plottingscripts/figures/lineplot/cum/')
    loss_columns = ['Full precision', 'Post minmaxQ', 'Post quantileQ', 'Pre minmaxQ', 'Pre quantileQ',
                    'HardQ', 'Bitwise hardQ', 'Bitwise quantileQ', 'Bitwise minmaxQ']


    layouts = ['squared', 'singlerow']
    for layout in layouts:
        if lineplot:
            make_lineplot(layout)
        if boxplot:
            make_boxplot(layout)
        if boxplotbestfold:
            make_boxplotbest(layout)



