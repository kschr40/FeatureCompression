# FeatureCompression


## Data 
The file results `results/processed_kFold_results\allhyperparameter.csv` contains the data for the best hyperparameter configuration for
each bit (`[2,3,4,5,6,7,8]` = 7), dataset (`['california', 'cpu_act', 'fried', 'sulfur', 'superconduct', 'wine']` = 6), and method tested (8 + Full Precision = 9).
The folder `results/processed_kFold_results\fulldata` accumulates the results for each dataset.
The folder `results/processed_kFold_results\avg_by_hyper` averages the MSE over all folds per dataset. 
In case of interest the processing script is located in `plottingscripts/process_kFolds_data.py`.
The script `plottingscripts/merge_and_remove_duplicates.py` was used to merge intermediate results.
The column `hyperparameter_setting_identifier` is not a unique identifier and changes when data is accumulated. 

 ## Plotting 
`plottingscripts/line_and_box_plot_cummulative.py` builds the plots listed in the paper. To view boxplots for 
different bits the y-limiter must be adjusted. It should be called from the root of the source code (`python3 plottingsscripts/line_and_boxplot_cummulative.py`)

