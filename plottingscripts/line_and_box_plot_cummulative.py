import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse
import seaborn as sns
import matplotlib.font_manager

pd.set_option('display.max_columns', None)
def create_boxplot_bits(ax, results : pd.DataFrame, name : str, loss_columns : [], hline : int = 0, ymax : float = None, ymin : float = 0):
    ax.boxplot(results[loss_columns], ylim=[ymin, ymax])
    ax.axhline(y=hline, linestyle='--', color=(1, 0, 0, 0.5))
    ax.set_title(name)

def initialize_figures(num_figures: int, layout: str, line: bool = False):
    if layout == 'singlerow':
        fig, axes = plt.subplots(1, num_figures, figsize=(7, 7/1.5), dpi=1000)
    elif layout == 'squared':
        half = num_figures//2
        if line == True:
            fig, axes = plt.subplots(2, half, figsize=(7, 4), dpi=1000, sharex=True)
        else:
            fig, axes = plt.subplots(2, half, figsize=(7, 4), dpi=1000)
    else:
        fig, axes = plt.subplots(1, num_figures, figsize=((7 * num_figures), 7/1.5), dpi=1000)
    return fig, axes

# Makes a lineplot for the best hyperparameter setup found. (Figure 4)
def make_lineplot(layout: str):
    counter = 0
    fig, axes = initialize_figures(num_figures_box, layout, True)
    methods = ['FP', 'Po-MQ', 'Po-QQ', 'Pr-MQ', 'Pr-QQ', 'Bw-MQ', 'Bw-QQ', 'SQ', 'Bw-SQ']
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
        values_per_bit = pd.DataFrame()
        for bit in bits:
            subset = dataframe[dataframe['bits'] == bit]
            minval_hyperparameter = subset[subset['FP'] == subset['FP'].min()]
            for method in methods:
                minval_hyperparameter[method] = subset[method].min()
            values_per_bit = pd.concat([values_per_bit, minval_hyperparameter], ignore_index=True)
        minimum = values_per_bit['FP'].mean()
        if data == 'california':
            realminimum = minimum - ((0.9 - minimum) * 0.2)
            ylim = (realminimum, 0.9)
        elif data == 'wine':
            realminimum = minimum - ((0.9 - minimum) * 0.2)
            ylim = (realminimum, 0.9)
        elif data == 'fried':
            realminimum = minimum - ((0.08 - minimum) * 0.2)
            ylim = (realminimum, 0.08)
        elif data == 'superconduct':
            realminimum = minimum - ((0.14 - minimum) * 0.2)
            ylim = (realminimum, 0.14)
        elif data == 'sulfur':
            realminimum = minimum - ((0.8 - minimum) * 0.2)
            ylim = (realminimum, 0.8)
        elif data == 'cpu_act':
            realminimum = minimum - ((0.08 - minimum) * 0.2)
            ylim = (realminimum, 0.08)
        else:
            ylim = (0, 5)
        leftoverlines = ['Po-MQ', 'Po-QQ']
        prlines = ['Pr-MQ', 'Pr-QQ']
        softq = ['Bw-SQ']
        colors = matplotlib.colormaps['tab10'].colors
        currentaxes.axhline(y=values_per_bit['FP'].mean(), linestyle='--', color='r', label='FP', lw=0.5)
        values_per_bit.plot(x='bits', y=leftoverlines, kind='line', linestyle=(0, (1, 4)), marker='o', ax=currentaxes,
                            legend=False, ylim=ylim, ms=1, lw=0.5)
        values_per_bit.plot(x='bits', y=prlines, kind='line', linestyle=(0, (1, 4)), marker='o', ax=currentaxes,
                            legend=False, ylim=ylim, ms=1, lw=0.5)
        values_per_bit.plot(x='bits', y=softq, kind='line', linestyle=(0, (3, 1)), marker='o', ax=currentaxes,
                            legend=False, ylim=ylim, ms=1, lw=0.5)
        currentaxes.set_xlabel('Bits')
        if (counter % 2) == 0 and counter // 2 == 0 or (counter % 2) == 1 and counter // 2 == 0:
            currentaxes.set_ylabel('MSE Loss')
        currentaxes.set_xticks(np.arange(2, 9, step=1))
        currentaxes.set_title(data)
        counter += 1
        plt.tight_layout(rect=[0, 0, 1, 1])
    fig.subplots_adjust(bottom=0.20, wspace=0.2)
    mylabels = ['FP', ' ', 'Po-MQ', 'Po-QQ', 'Pr-MQ', 'Pr-QQ', 'SQ (ours)', 'Bw-MQ (ours)', 'Bw-QQ (ours)', 'Bw-SQ (ours)']
    leg = axes[1, 1].legend(loc='upper center', labels=mylabels,
                      bbox_to_anchor=(0.5, -0.25), fancybox=False, shadow=False, ncol=5)
    for i, text in enumerate(leg.get_texts()):
        if i in [6,7,8,9]:
            text.set_weight("bold")

    fig.savefig(f'plottingscripts/figures/lineplot/cum/lineplot_{layout}.png')
    fig.savefig(f'plottingscripts/figures/lineplot/cum/lineplot_{layout}.pdf')
    #plt.show()
    plt.close(fig)

def calc_hyperparameter():
    methods = ['FP', 'Po-MQ', 'Po-QQ', 'Pr-MQ', 'Pr-QQ', 'Bw-MQ', 'Bw-QQ', 'SQ', 'Bw-SQ']

    hyperparamcolumns = ['dataset', 'hyperparameter_setting_id', 'bits', 'weight_decay', 'learning_rate',
                         'hidden_layers', 'hidden_neurons', 'num_epochs', 'decrease_factor', 'method', 'min']
    fewhyperparamcolumns = ['hyperparameter_setting_id', 'bits', 'weight_decay', 'learning_rate',
                         'hidden_layers', 'hidden_neurons', 'num_epochs', 'decrease_factor']
    all_hyperparameter = pd.DataFrame(columns=hyperparamcolumns)

    for data in datasets:
        path = f'results/processed_kFold_results/avg_by_hyper/' + f'{data}_hyperparameter.csv'
        dataframe = pd.read_csv(path, sep=',', index_col=False, on_bad_lines='warn')
        bits = [2, 3, 4, 5, 6, 7, 8]
        values_per_bit = pd.DataFrame()
        for bit in bits:
            subset = dataframe[dataframe['bits'] == bit]
            minval_hyperparameter = subset[subset['FP'] == subset['FP'].min()]
            collect = pd.DataFrame(columns=hyperparamcolumns)
            for method in methods:
                min_index = subset[method].idxmin()
                minval_hyperparameter[method] = subset[method].min()
                checkdf = subset.loc[[min_index], fewhyperparamcolumns]
                checkdf['method'] = method
                checkdf['dataset'] = data
                checkdf['min'] = minval_hyperparameter[method].item()
                collect = pd.concat([collect, checkdf])
            values_per_bit = pd.concat([values_per_bit, minval_hyperparameter], ignore_index=True)
            all_hyperparameter = pd.concat([all_hyperparameter, collect])
    all_hyperparameter.to_csv("results/processed_kFold_results/allhyperparameter.csv", index=False)
    all_hyperparameter.to_latex("results/processed_kFold_results/allhyperparameter.tex", index=False)

# Makes a boxplot over all data points collected.
def make_boxplot(layout: str):
    bits = [2, 3, 4, 5, 6, 7, 8]
    for bit in bits:
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
            dfsubset4 = dataframe[dataframe['bits'] == bit]
            hline = dfsubset4['FP'].median()
            subdata = dfsubset4[loss_columns]
            boxplotlist = []
            for loss_colum in loss_columns:
                name_list = subdata[loss_colum].tolist()
                boxplotlist.append(name_list)
            ax.boxplot(boxplotlist, positions=[0,1,2,3,4,5,6,7,8], labels=loss_columns,
                           boxprops=dict(ms=1, linewidth=0.3, color='black'),
                           whiskerprops=dict(ms=1, linestyle='-', linewidth=0.3, color='black'),
                           flierprops=dict(ms=1, linewidth=0.3),
                           capprops=dict(ms=1, linewidth=0.3, color='black'))
            sns.swarmplot(data=boxplotlist, ax=ax, color='black', alpha=0.5, linewidth=0.5, size=1, orient='v', dodge=True)

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
                ymin = 0
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
                          labels=loss_columns, rotation=90)
        fig.tight_layout()
        fig.savefig(f'plottingscripts/figures/boxplot/{bit}_{layout}_all_boxplot.pdf', bbox_inches='tight')
        plt.close(fig)

# Makes a boxplot for the best hyperparameter setup found. (Figure 5 - 4bit)
def make_boxplotbest(layout:str):
    bits = [2, 3, 4, 5, 6, 7, 8]
    for bit in bits:
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
            methods = ['FP', 'Po-MQ', 'Po-QQ', 'Pr-MQ', 'Pr-QQ', 'Bw-MQ', 'Bw-QQ', 'SQ', 'Bw-SQ']
            values_per_bit = pd.DataFrame()
            hyperparameters = {}
            subset = avgdataframe[avgdataframe['bits'] == bit]
            for method in methods:
                item = subset[subset[method] == subset[method].min()]
                hyperparameters[method] = item['hyperparameter_setting_id'].values[0]
            subset = fulldataframe[fulldataframe['bits'] == bit]
            prefered_hyperparameter = pd.DataFrame()
            for method in methods:
                if prefered_hyperparameter.empty:
                    prefered_hyperparameter = subset[subset['hyperparameter_setting_id'] == hyperparameters[method]]
                    continue
                dfbest = subset[subset['hyperparameter_setting_id'] == hyperparameters[method]][method]
                prefered_hyperparameter[method] = dfbest.values
            values_per_bit = pd.concat([values_per_bit, prefered_hyperparameter], ignore_index=True)

            hline = values_per_bit['FP'].median()
            ax.axhline(y=hline, linestyle='--', color=(1, 0, 0, 0.5), lw=0.5)
            boxplotlist = []
            loss_columns = ['FP', 'Po-MQ', 'Po-QQ', 'Pr-MQ', 'Pr-QQ', 'SQ', 'Bw-MQ', 'Bw-QQ', 'Bw-SQ']
            swarmplotdf = pd.DataFrame(columns=loss_columns)
            for loss_colum in loss_columns:
                name_list = values_per_bit[loss_colum].tolist()
                swarmplotdf[loss_colum] = values_per_bit[loss_colum]
                boxplotlist.append(name_list)

            ax.set_title(f'{data}')
            ax.boxplot(boxplotlist, positions=[0,1,2,3,4,5,6,7,8], tick_labels=loss_columns,
                       boxprops=dict(ms=1, linewidth=0.3, color='black'),
                       whiskerprops=dict(ms=1, linestyle='-', linewidth=0.3, color='black'),
                       flierprops=dict(ms=1, linewidth=0.3),
                       capprops=dict(ms=1, linewidth=0.3, color='black'))
            sns.swarmplot(data=swarmplotdf, ax=ax, color='black', alpha=0.5, linewidth=0.5, size=1, orient='v', dodge=True)

            ax.set_ylabel(None)
            if data == 'cpu_act':
                if bit == 2:
                    ymin = 0.015
                    ymax = 0.05
                if bit in [5,6,7]:
                    ymin = 0.014
                    ymax = 0.030
                if bit in [3,4]:
                    ymin = 0.015
                    ymax = 0.04
                if bit in [2,3,4,5,6,7]:
                    ax.set_ylim([ymin, ymax])
            if data == 'sulfur' and bit == 2:
                ymin = 0.015
                ymax = 0.8
                ax.set_ylim([ymin, ymax])
            if data == 'superconduct':
                if bit == 2:
                    ymin = 0.06
                    ymax = 0.15
                if bit == 6:
                    ymin = 0.06
                    ymax = 0.11
                if bit in [2,6]:
                    ax.set_ylim([ymin, ymax])
            if data == 'fried':
                    if bit == 2:
                        ymin = 0.03
                        ymax = 0.18
                    if bit == 6:
                        ymin = 0.04
                        ymax = 0.052
                    if bit in [2,6]:
                        ax.set_ylim([ymin, ymax])
            if data == 'california':
                    if bit == 2:
                        ymin = 0.2
                        ymax = 0.8
                    if bit in [5,6,7,8]:
                        ymin = 0.27
                        ymax = 0.5
                    if bit in [2, 5, 6, 7,8]:
                        ax.set_ylim([ymin, ymax])
            if (counter % 2) == 0 and counter // 2 == 0 or (counter % 2) == 1 and counter // 2 == 0:
                ax.set_ylabel('MSE Loss')
            counter += 1
            ax.yaxis.grid(True)

            ax.set_xticks([y for y in range(len(loss_columns))],
                          labels=['FP', 'Po-MQ', 'Po-QQ', 'Pr-MQ', 'Pr-QQ', 'SQ', 'Bw-MQ', 'Bw-QQ', 'Bw-SQ'], rotation=90)
            labels = ax.get_xticklabels()
            for i, text in enumerate(labels):
                if i in [5,6,7,8]:
                     text.set_weight("bold")
            fig.tight_layout()
        fig.savefig(f'plottingscripts/figures/boxplot/{bit}_{layout}_boxplot_kfoldsbest.pdf', bbox_inches='tight')

        plt.close(fig)

if __name__ == "__main__":

    print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
    parser = argparse.ArgumentParser(description="Process some input arguments.")
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

    datasets = ['california', 'cpu_act', 'fried', 'sulfur', 'superconduct', 'wine']
    boxdatasets = ['california', 'cpu_act', 'fried', 'sulfur', 'superconduct', 'wine']
    num_figures = len(datasets)
    num_figures_box = len(boxdatasets)

    font = {'family' : 'Times New Roman',
            'size'   : 8}

    plt.rc('font', **font)
    if boxplot or boxplotbestfold:
        if not os.path.exists('plottingscripts/figures/boxplot/cum/'):
            os.makedirs('plottingscripts/figures/boxplot/cum/')
    if lineplot:
        if not os.path.exists('plottingscripts/figures/lineplot/cum/'):
            os.makedirs('plottingscripts/figures/lineplot/cum/')
    loss_columns = ['FP', 'Po-MQ', 'Po-QQ', 'Pr-MQ', 'Pr-QQ', 'Bw-MQ', 'Bw-QQ', 'SQ', 'Bw-SQ']

    layouts = ['squared'] #, 'singlerow'
    for layout in layouts:
        if lineplot:
            make_lineplot(layout)
        if boxplot:
            make_boxplot(layout)
        if boxplotbestfold:
            make_boxplotbest(layout)



