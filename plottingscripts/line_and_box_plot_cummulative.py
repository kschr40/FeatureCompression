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
            fig, axes = plt.subplots(2, half, figsize=(8.5, 4), dpi=1000, sharex=True)
        else:
            fig, axes = plt.subplots(2, half, figsize=(7, 4), dpi=1000)
    else:
        fig, axes = plt.subplots(1, num_figures, figsize=((7 * num_figures), 7/1.5), dpi=1000)
    return fig, axes

# Makes a lineplot for the best hyperparameter setup found. (Figure 4)
def make_lineplot(layout: str):
    counter = 0
    fig, axes = initialize_figures(num_figures_box, layout, True)
    for data in datasets:
        if layout == 'squared':
            currentaxes = axes[counter % 2, counter // 2]
        elif layout == 'singlerow':
            currentaxes = axes[counter]
        else:
            currentaxes = axes[counter]
        path = f'results/best_hyperparameter_multiple/{data}_best_hyperparameter_test_multiple.csv'
        dataframe = pd.read_csv(path, sep=',', index_col=False, on_bad_lines='warn')
        bits = [2, 3, 4, 5, 6, 7, 8]
        values_per_bit = pd.DataFrame()
        methods = ['FP', 'Pr-MQ', 'Pr-QQ', 'LLT4', 'LLT9', 'LSQ', 'Bw-SQ']
        for bit in bits:
            subset = dataframe[dataframe['bits'] == bit]
            minval_hyperparameter = subset[subset['val_loss_FP'] == subset['val_loss_FP'].min()]
            for method in methods:
                minval_hyperparameter[f'val_loss_{method}'] = minval_hyperparameter[f'test_loss_{method}'].mean()
            values_per_bit = pd.concat([values_per_bit, minval_hyperparameter], ignore_index=True)
        if data == 'california':
            ylim = (0.12, 0.6)
        elif data == 'wine_quality':
            ylim = (0.501, 0.75)
        elif data == 'fried':
            ylim = (0.04, 0.08)
        elif data == 'superconduct':
            ylim = (0.075, 0.14)
        elif data == 'sulfur':
            ylim = (0.08, 0.62)
        elif data == 'cpu_act':
            ylim = (0.015, 0.05)
        else:
            ylim = (0, 5)
        leftoverlines = ['val_loss_LLT4', 'val_loss_LLT9', 'val_loss_LSQ']
        prlines = ['val_loss_Pr-MQ', 'val_loss_Pr-QQ']
        softq = ['val_loss_Bw-SQ']
        colors = matplotlib.colormaps['tab20'].colors
        mycolorlist = [colors[12],colors[13]]
        mycolorlistcomparison = [colors[0],colors[1],colors[2]]
        currentaxes.axhline(y=values_per_bit['val_loss_FP'].mean(), linestyle='--', color='r', label='FP', lw=0.8)
        values_per_bit.plot(x='bits', y=prlines, kind='line', linestyle=(0, (3, 1)), marker='^', ax=currentaxes,
                            legend=False, ylim=ylim, ms=1, lw=0.8, color=mycolorlist)
        values_per_bit.plot(x='bits', y=leftoverlines, kind='line', linestyle=(0, (3, 1)), marker='v', ax=currentaxes,
                            legend=False, ylim=ylim, ms=1, lw=0.8, color=mycolorlistcomparison)
        values_per_bit.plot(x='bits', y=softq, kind='line', marker='o', ax=currentaxes,
                            legend=False, ylim=ylim, ms=1, lw=0.8, color=colors[4])
        currentaxes.set_xlabel('Bits')
        if (counter % 2) == 0 and counter // 2 == 0 or (counter % 2) == 1 and counter // 2 == 0:
            currentaxes.set_ylabel('MSE Loss')
        currentaxes.set_xticks(np.arange(2, 9, step=1))
        currentaxes.set_title(data)
        counter += 1
        plt.tight_layout(rect=[0, 0, 1, 1])
    fig.subplots_adjust(bottom=0.20, wspace=0.2)
    mylabels = ['FP', 'Pr-MQ', 'Pr-QQ', 'LLT4', 'LLT9', 'LSQ', 'Bw-SQ (ours)']
    leg = axes[1, 1].legend(loc='upper center', labels=mylabels,
                      bbox_to_anchor=(0.5, -0.25), fancybox=False, shadow=False, ncol=7)
    for i, text in enumerate(leg.get_texts()):
        if i in [6]:
            text.set_weight("bold")

    fig.savefig(f'plottingscripts/figures/lineplot/lineplot_{layout}.png')
    fig.savefig(f'plottingscripts/figures/lineplot/lineplot_{layout}.pdf')
    #plt.show()
    plt.close(fig)

# Makes a boxplot over all data points collected.
def make_boxplot(layout: str):
    bits = [2, 3, 4, 5, 6, 7, 8]
    loss_columns_test = ['test_loss_FP', 'test_loss_Pr-MQ', 'test_loss_Pr-QQ', 'test_loss_LLT4', 'test_loss_LLT9', 'test_loss_LSQ', 'test_loss_Bw-SQ']

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

            path = f'results/best_hyperparameter_multiple/{data}_best_hyperparameter_test_multiple.csv'
            dataframe = pd.read_csv(path, sep=',', index_col=False, on_bad_lines='warn')
            dfsubset4 = dataframe[dataframe['bits'] == bit]
            hline = dfsubset4['test_loss_FP'].median()
            subdata = dfsubset4[loss_columns_test]
            boxplotlist = []
            for loss_colum in loss_columns_test:
                name_list = subdata[loss_colum].tolist()
                boxplotlist.append(name_list)
            ax.boxplot(boxplotlist, positions=[0,1,2,3,4,5,6],
                           boxprops=dict(ms=1, linewidth=0.3, color='black'),
                           whiskerprops=dict(ms=1, linestyle='-', linewidth=0.3, color='black'),
                           flierprops=dict(ms=1, linewidth=0.3),
                           capprops=dict(ms=1, linewidth=0.3, color='black'))
            sns.swarmplot(data=boxplotlist, ax=ax, color='black', alpha=0.5, linewidth=0.5, size=1, orient='v', dodge=True)

            if bit == 2:
                ymin = None
                if data == 'california':
                    ymax = 0.65
                    ymin = 0.1
                elif data == 'wine':
                    ymax = 1
                    ymin = 0.4
                elif data == 'fried':
                    ymax = 0.25
                    ymin = 0.01
                elif data == 'superconduct':
                    ymax = 0.25
                elif data == 'sulfur':
                    ymax = 0.9
                    ymin = 0.05
                elif data == 'cpu_act':
                    ymax = 0.05
                    ymin = 0.01
                else:
                    ymax = None
                ax.set_ylim([ymin, ymax])
            if bit == 3 or bit == 4:
                ymin = None
                if data == 'cpu_act':
                    ymax = 0.05
                    ax.set_ylim([ymin, ymax])

            ax.axhline(y=hline, linestyle='--', color=(1, 0, 0, 0.5))
            ax.set_title(f'{data}')
            ax.set_ylabel(None)
            if counter % 2 == 0:
                ax.set_xticks([y for y in range(len(loss_columns_test))],
                              labels=[])
            else:
                ax.set_xticks([y for y in range(len(loss_columns_test))],
                              labels=labels, rotation=90)
                ax.get_xticklabels()[-1].set_fontweight('bold')
            if (counter % 2) == 0 and counter // 2 == 0 or (counter % 2) == 1 and counter // 2 == 0:
                ax.set_ylabel('MSE Loss')
            counter += 1
            ax.yaxis.grid(True)

        fig.tight_layout()
        fig.savefig(f'plottingscripts/figures/boxplot/{bit}_{layout}_all_boxplot.pdf', bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument('--lineplot', action='store_true', help='Create lineplot')
    parser.add_argument('--no-lineplot', action='store_false', dest='lineplot', help='Not create lineplot')
    parser.add_argument('--boxplot', action='store_true', help='Create boxplot')
    parser.add_argument('--no-boxplot', action='store_false', dest='boxplot', help='Not create boxplot')
    parser.set_defaults(lineplot=True)
    parser.set_defaults(boxplot=False)
    parser.set_defaults(boxplotbestfold=True)

    args = parser.parse_args()

    lineplot = args.lineplot
    boxplot = args.boxplot
    boxplotbestfold = args.boxplotbestfold

    datasets = ['california', 'cpu_act', 'fried', 'sulfur', 'superconduct', 'wine_quality']
    boxdatasets = ['california', 'cpu_act', 'fried', 'sulfur', 'superconduct', 'wine_quality']
    num_figures = len(datasets)
    num_figures_box = len(boxdatasets)

    font = {'family' : 'Times New Roman',
            'size'   : 8}

    plt.rc('font', **font)
    if boxplot or boxplotbestfold:
        if not os.path.exists('plottingscripts/figures/boxplot/'):
            os.makedirs('plottingscripts/figures/boxplot/')
    if lineplot:
        if not os.path.exists('plottingscripts/figures/lineplot/'):
            os.makedirs('plottingscripts/figures/lineplot/')
    labels = ['FP', 'Pr-MQ', 'Pr-QQ', 'LLT4', 'LLT9', 'LSQ', 'Bw-SQ']
    loss_columns = ['test_loss_FP', 'test_loss_Pr-MQ', 'test_loss_Pr-QQ', 'test_loss_LLT4', 'test_loss_LLT9', 'test_loss_LSQ', 'test_loss_Bw-SQ']

    layouts = ['squared'] #, 'singlerow'
    for layout in layouts:
        if lineplot:
            make_lineplot(layout)
        if boxplot:
            make_boxplot(layout)



