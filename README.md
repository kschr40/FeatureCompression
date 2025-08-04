# FeatureCompression


## Data 
The file results `results/processed_kFold_results\allhyperparameter.csv` contains the data for the best hyperparameter configuration for
each bit (`[2,3,4,5,6,7,8]` = 7), dataset (`[ "california",  "cpu_act",  "fried",  "sulfur",  "superconduct",  "wine_quality"]` = 6), and method tested (8 + Full Precision = 9).
The folder `results/processed_kFold_results\fulldata` accumulates the results for each dataset.
The folder `results/processed_kFold_results\avg_by_hyper` averages the MSE over all folds per dataset. 
In case of interest the processing script is located in `plottingscripts/process_kFolds_data.py`.
The script `plottingscripts/merge_and_remove_duplicates.py` was used to merge intermediate results.
The column `hyperparameter_setting_identifier` is not a unique identifier and changes when data is accumulated. 

## Plotting 
`plottingscripts/line_and_box_plot_cummulative.py` builds the plots listed in the paper. To view boxplots for 
different bits the y-limiter must be adjusted. It should be called from the root of the source code (`python3 plottingsscripts/line_and_boxplot_cummulative.py`)

## Reproduction
Other software versions might also work but are not tested. Only compatible with UNIX Systems.
1. load GCC (12.3.0) 
2. load PyTorch 2.1.2 for CUDA 12.1 (recommended - without CUDA also works)
3. Use your favorite tool to execute multiple experiments (e.g. bash with bit as parameter). Note that experiments might run for several days for a bit and single dataset configuration. In case you want to test a small run add the `--debug` flag (reduces the hidden_neurons to 10). 
```bash
datasets=("california" "cpu_act"  "fried"  "sulfur"  "superconduct"  "wine_quality")
num_bits=$4
timestamp=$(date +%s)

for dataset in "${datasets[@]}"; do
python3 hyperparameter_tuning.py --dataset $dataset --n_bits $num_bits --n_steps 50 --scratch {folder where data can be stored} --result_folder {folder where results should be stored} > {file for output}
done
```
4. Results are stored for each step, with the naming scheme `{datasetname}_hyperparameter_tuning_{nbits}bits_{stepsdone}steps.csv`