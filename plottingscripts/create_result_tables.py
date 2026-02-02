import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datasets = ['california', 'fried', 'superconduct', 'cpu_act', 'sulfur', 'wine_quality']
dataset_to_latex = {
    'california': '\\california',
    'fried': "\\" + 'fried',
    'superconduct': '\\superconduct',
    'cpu_act': '\\cpu',
    'sulfur': '\\sulfur',
    'wine_quality': '\\wine'
}


# from pandas.io.formats.latex import LatexFormatter

def val_to_str(val, min_val):
    """Format the value for LaTeX output, bolding the minimum value."""
    if pd.isnull(val):
        return 'NaN'
    elif val == min_val:
        return f"\\textbf{{{val:.3f}}}"
    elif val <= min_val * 1.025:
        return f"\\underline{{{val:.3f}}}"
    else:
        return f"{val:.3f}"

def bold_min_in_row(df):
    def formatter(row):
        # Only gets quantization columns
        min_val = row.min()
        # return [f"\\textbf{{{v:.3f}}}" if v == min_val and pd.notnull(v) else f"{v:.3f}" for v in row]
        return [val_to_str(v, min_val) for v in row]
    formatted = df.copy()
    formatted.iloc[:, 2:] = formatted.iloc[:, 2:].astype('float')
    formatted.iloc[:,2] = formatted.iloc[:,2].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else 'NaN')
    formatted.iloc[:, 3:] = df.iloc[:, 3:].apply(formatter, axis=1, result_type='expand')
    return formatted

def to_latex_multirow(df, dataset_col='dataset', bits_col='bits', use_FP = True, n_comparison = 5, n_ours = 1):
    # Prepare the DataFrame for multirow
    df = df.copy()
    df[dataset_col] = df[dataset_col].replace('cpu_act', 'cpu\\_act')
    df[dataset_col] = df[dataset_col].replace('wine_quality', 'wine\\_quality')
    lines = []
    n_rows = df.shape[0]
    dataset_counts = df[dataset_col].value_counts()[df[dataset_col].unique()].tolist()
    idx = 0
    for dataset, count in zip(df[dataset_col].unique(), dataset_counts):
        for i in range(count):
            row = df.iloc[idx]
            row_str = []
            if i == 0:
                row_str.append(f"\\multirow{{{count}}}{{*}}{{{dataset}}}")
            else:
                row_str.append("")
            row_str.append(str(row[bits_col]))
            for col in df.columns[2:]:
                row_str.append(str(row[col]))
            lines.append(" & ".join(row_str) + " \\\\")
            idx += 1
        lines.append("\\hline")
    # Build the LaTeX table
    if use_FP:
        first_header = "\\multirow{2}{*}{dataset} & \\multirow{2}{*}{bits} & \\multirow{2}{*}{FP} & \\multicolumn{" + str(n_comparison) + "}{c|}{Comparison models} & \\multicolumn{" + str(n_ours) + "}{c}{Ours}  \\\\"
        header = " & ".join(['','',''] + list(df.columns[3:])) + " \\\\"
        latex = "\\begin{tabular}{lcc|" + "c" * (n_comparison) + '|' + "c" * (n_ours)  + "}\n\\toprule\n"
    else: 
        first_header = "\\multirow{2}{*}{dataset} & \\multirow{2}{*}{bits} & \\multicolumn{" + str(n_comparison) + "}{c|}{Comparison models} & \\multicolumn{" + str(n_ours) + "}{c}{Ours}  \\\\"
        header = " & ".join(['',''] + list(df.columns[2:])) + " \\\\"
        latex = "\\begin{tabular}{lc|" + "c" * (n_comparison) + '|' + "c" * (n_ours)  + "}\n\\toprule\n"
    
    latex += first_header
    latex += header + "\n\\midrule\n"
    latex += "\n".join(lines)
    latex += "\n\\bottomrule\n\\end{tabular}"
    return latex

def preprocessing_df(df, bits = 4, methods = ['FP', 'Pr-MQ', 'Pr-QQ',  'LLT4', 'LLT9', 'LSQ', 'Bw-SQ']):
    test_loss_col = [col for col in df.columns if 'test_loss' in col]
    test_loss_col_short = [col.replace('test_loss_', '') for col in test_loss_col]
    df = df[df['bits'] == bits]
    df_short = df[test_loss_col]
    df_short.columns = test_loss_col_short
    df_short = df_short[methods]
    return df_short

import scipy.stats as stats
def calculate_confidence_intervals(df, confidence_level=0.95):
    confidence_intervals = {}
    mean_values = df.mean()
    for column in df.columns:
        # Sample statistics
        sample_mean = df[column].mean()
        sample_std = df[column].std(ddof=1)
        n = len(df[column])
        
        # Degrees of freedom
        df_deg = n - 1
        
        # Critical value from t-distribution
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df_deg)
        
        # Margin of error
        margin_of_error = t_critical * (sample_std / np.sqrt(n))
        
        # Confidence interval
        confidence_intervals[column] = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    confidence_interval_df = pd.DataFrame.from_dict(confidence_intervals, orient='index', columns=['Lower Bound', 'Upper Bound'])
    confidence_interval_df = pd.concat([confidence_interval_df, mean_values.rename('Mean')], axis=1)
    confidence_interval_df.index.name = 'Method'
    confidence_interval_df = confidence_interval_df.iloc[:,[0,2,1]]
    confidence_interval_df['Confidence Interval'] = confidence_interval_df.apply(
    lambda row: f"[{row['Lower Bound']:.3f}, {row['Upper Bound']:.3f}]", axis=1
)
    confidence_interval_df = confidence_interval_df[['Mean', 'Confidence Interval']]
    return confidence_interval_df

def get_best_avg_results(methods, datasets, result_dir):
    best_avg_results = pd.DataFrame()
    for dataset in datasets:
        df_path = f"{result_dir}/{dataset}_best_hyperparameter_test_multiple.csv"
        for bits in range(2,9):
            df = pd.read_csv(df_path)    
            df_short = preprocessing_df(df, bits=bits, methods=methods)
            current_dict = {'dataset': dataset, 'bits': bits}
            current_dict.update(df_short.mean().to_dict())
            current_df = pd.DataFrame(current_dict, index=[0])
            best_avg_results = pd.concat([best_avg_results, current_df], ignore_index=True)
    return best_avg_results

if __name__ == "__main__":
    
    ## Save best average results for FP, Baseline models and Bw-SQ
    methods = ['FP', 'Pr-MQ', 'Pr-QQ',  'LLT4', 'LLT9', 'LSQ', 'Bw-SQ']
    result_dir = "../results/best_hyperparameter_multiple"
    best_avg_results = get_best_avg_results(methods, datasets, result_dir)
    
    best_avg_results['dataset'] = best_avg_results['dataset'].replace(dataset_to_latex)

    best_avg_results_bold = bold_min_in_row(best_avg_results)
    best_avg_results_bold.to_latex(
        buf='../results/tables/best_avg_results_bold.tex',
        index=False,
        escape=False,
        caption='Average MSE Loss for selected hyperparameter setting (bold: minimal per row)',
        label='tab:best_avg_results_bold',
        float_format="%.3f"
    );

    latex_multirow = to_latex_multirow(best_avg_results_bold)
    with open('../results/tables/best_avg_results_bold_multirow_it.tex', 'w') as f:
        f.write(latex_multirow);
        
    ## Ablation study tables
    methods = ['SQ', 'Bw-MQ', 'Bw-QQ', 'Bw-SQ']
    result_dir = "../results/best_hyperparameter_multiple"
    best_avg_results = get_best_avg_results(methods, datasets, result_dir)

    best_avg_results['dataset'] = best_avg_results['dataset'].replace(dataset_to_latex)


    best_avg_results_bold = bold_min_in_row(best_avg_results)
    best_avg_results_bold.to_latex(
        buf='../results/tables/best_avg_results_bold.tex',
        index=False,
        escape=False,
        caption='Average MSE Loss for selected hyperparameter setting (bold: minimal per row)',
        label='tab:best_avg_results_bold',
        float_format="%.3f"
    )

    latex_multirow = to_latex_multirow(best_avg_results_bold, use_FP=False, n_comparison=3)
    with open('../results/tables/ablation_best_avg_results_bold_multirow_it.tex', 'w') as f:
        f.write(latex_multirow)
        
    best_avg_results.rename(columns={'dataset': 'Dataset'}, inplace=True)
    best_avg_result_per_dataset = best_avg_results.groupby('Dataset').mean()
    best_avg_result_per_dataset = best_avg_result_per_dataset.reindex([dataset_to_latex[dataset] for dataset in datasets])
    best_avg_result_per_dataset = best_avg_result_per_dataset[methods]
    best_avg_result_per_dataset[methods[:-1]] = best_avg_result_per_dataset[methods[:-1]].values / best_avg_result_per_dataset[methods[-1]].values.reshape(-1,1)
    ours_row = best_avg_result_per_dataset[methods[-1]]
    best_avg_result_per_dataset = best_avg_result_per_dataset.drop(columns=[methods[-1]])

    ours_row = pd.DataFrame(ours_row).T
    ours_row['Mean'] = ours_row.mean(axis=1)
    ours_row = ours_row.applymap(lambda x: f"${x:.3f}$")
        
        
    percentage_results = (best_avg_result_per_dataset * 100 - 100).T
    percentage_results['Mean'] = percentage_results.mean(axis=1)
    percentage_results = percentage_results[[dataset_to_latex[dataset] for dataset in datasets] + ['Mean']]
    percentage_results.rename(columns=dataset_to_latex, inplace=True)
    pos_mark = percentage_results >= 0
    percentage_results = percentage_results.astype(float)
    percentage_results = percentage_results.abs().map(lambda x: f"{x:.2f} \%")
    percentage_results[pos_mark] = percentage_results[pos_mark].map(lambda x: f"$+{x}$")
    percentage_results[~pos_mark] = percentage_results[~pos_mark].map(lambda x: f"$-{x}$")
    percentage_results = pd.concat([ours_row, percentage_results])

    percentage_results.to_latex('../results/tables/ablation_short.tex')   
    
    
    ## Calculate confidence intervals
    methods = ['FP', 'Pr-MQ', 'Pr-QQ',  'LLT4', 'LLT9', 'LSQ', 'Bw-SQ']
    result_dir = "../results/best_hyperparameter_multiple"
    conf_interval_df = pd.DataFrame()
    for dataset in datasets:
        df_path = f"{result_dir}/{dataset}_best_hyperparameter_test_multiple.csv"
        for bits in range(2,9):
            df = pd.read_csv(df_path)    
            df_short = preprocessing_df(df, bits=bits, methods=methods)
            confidence_interval_df = calculate_confidence_intervals(df_short)
            current_dict = {'dataset': dataset, 'bits': bits}
            current_dict.update(confidence_interval_df['Confidence Interval'])
            current_df = pd.DataFrame(current_dict, index=[0])
            conf_interval_df = pd.concat([conf_interval_df, current_df], ignore_index=True) 
    latex_multirow_conf = to_latex_multirow(conf_interval_df)
    with open('../results/tables/conf_interval_df_multirow.tex', 'w') as f:
        f.write(latex_multirow_conf)
