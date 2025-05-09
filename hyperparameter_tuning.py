from datasets import get_dataloader, get_min_max_values, get_quantization_thresholds
import torch
import pandas as pd
import random
from tqdm import tqdm
from training import train_mlp_model, train_mlp_pre_model, train_soft_mlp, train_soft_comp_mlp

def random_search_soft_quantization_threshold(train_loader, val_loader, n_steps = 10, n_bits =8, num_features=8, optimize_dict = {}, device = 'cpu'):
    thresholds = get_quantization_thresholds(train_loader, n_bits)
    min_values, max_values = get_min_max_values(train_loader, num_features=num_features)
    
    # Define default hyperparameters
    weight_decay =  0
    learning_rate = 0.001
    hidden_layers = 2
    hidden_neurons = 256
    num_epochs = 30
    add_noise = False
    decrease_factor = 0.001

    # Lists to store results
    val_loss_mlp_values = []
    val_loss_hard_post_mlp_values = []
    val_loss_hard_thr_post_mlp_values = []
    val_loss_hard_pre_mlp_values = []
    val_loss_hard_thr_pre_mlp_values = []
    val_loss_soft_mlp_values = []
    val_loss_soft_hard_mlp_values = []
    val_loss_soft_comp_mlp_values = []
    val_loss_soft_hard_comp_mlp_values = []
    

    hyperparameter_dict = {
        'weight_decay': [],
        'learning_rate': [],
        'hidden_layers': [],
        'hidden_neurons': [],
        'num_epochs': [],
        'decrease_factor': []}
    

    # Perform random search
    for _ in tqdm(range(n_steps)):
        for key, value in optimize_dict.items():
            if key == 'weight_decay':
                weight_decay = random.choice(value)
            elif key == 'learning_rate':
                learning_rate = random.choice(value)
            elif key == 'hidden_layers':
                hidden_layers = random.choice(value)
            elif key == 'hidden_neurons':
                hidden_neurons = random.choice(value)    
            elif key == 'num_epochs':
                num_epochs = random.choice(value)    
            elif key == 'add_noise':
                add_noise = random.choice(value)    
            elif key == 'decrease_factor':
                decrease_factor = random.choice(value)
            else:
                raise ValueError(f"Unknown hyperparameter: {key}")
            
        architecture = [8] + [hidden_neurons] * hidden_layers + [1]
        hyperparameter_dict['weight_decay'].append(weight_decay)
        hyperparameter_dict['learning_rate'].append(learning_rate)
        hyperparameter_dict['hidden_layers'].append(hidden_layers)
        hyperparameter_dict['hidden_neurons'].append(hidden_neurons)
        hyperparameter_dict['num_epochs'].append(num_epochs)
        hyperparameter_dict['decrease_factor'].append(decrease_factor)

        # Calculate losses for mlp model
        val_loss_mlp, val_loss_hard_post_mlp, val_loss_hard_thr_post_mlp = train_mlp_model(train_loader=train_loader, val_loader=val_loader,
            architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, device=device)
        
        # Calculate losses for pre-training quantization model
        val_loss_hard_pre_mlp, val_loss_hard_thr_pre_mlp = train_mlp_pre_model(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, device=device)

        # Calculate losses for quantization model
        val_loss_soft_mlp, val_loss_soft_hard_mlp = train_soft_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, decrease_factor=decrease_factor, device=device)

        # Calculate losses for quantization model
        val_loss_soft_comp_mlp, val_loss_soft_hard_comp_mlp = train_soft_comp_mlp(train_loader=train_loader, val_loader=val_loader, architecture=architecture, min_values=min_values, max_values=max_values, thresholds=thresholds,
            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            n_bits=n_bits, decrease_factor=decrease_factor, device=device)
        

        val_loss_mlp_values.append(val_loss_mlp)
        val_loss_hard_post_mlp_values.append(val_loss_hard_post_mlp)
        val_loss_hard_thr_post_mlp_values.append(val_loss_hard_thr_post_mlp)
        val_loss_hard_pre_mlp_values.append(val_loss_hard_pre_mlp)
        val_loss_hard_thr_pre_mlp_values.append(val_loss_hard_thr_pre_mlp)
        val_loss_soft_mlp_values.append(val_loss_soft_mlp)
        val_loss_soft_hard_mlp_values.append(val_loss_soft_hard_mlp)
        val_loss_soft_comp_mlp_values.append(val_loss_soft_comp_mlp)
        val_loss_soft_hard_comp_mlp_values.append(val_loss_soft_hard_comp_mlp)
 
    losses_df = pd.DataFrame({
        'val_loss_mlp': val_loss_mlp_values,
        'val_loss_hard_post_mlp': val_loss_hard_post_mlp_values,
        'val_loss_hard_thr_post_mlp': val_loss_hard_thr_post_mlp_values,
        'val_loss_hard_pre_mlp': val_loss_hard_pre_mlp_values,
        'val_loss_hard_thr_pre_mlp': val_loss_hard_thr_pre_mlp_values,
        'val_loss_soft_mlp': val_loss_soft_mlp_values,
        'val_loss_soft_hard_mlp': val_loss_soft_hard_mlp_values,
        'val_loss_soft_comp_mlp': val_loss_soft_comp_mlp_values,
        'val_loss_soft_hard_comp_mlp': val_loss_soft_hard_comp_mlp_values
    })

    # Create DataFrame with results
    results_df = pd.DataFrame(hyperparameter_dict)
    results_df = pd.concat([results_df, losses_df], axis=1)

    results_df = results_df.sort_values('val_loss_mlp')  # Sort by loss ascending    
    return results_df


if __name__ == "__main__":

    dataset = 'California_Housing'
    train_loader, val_loader, test_loader = get_dataloader(dataset = dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimize_dict = {'weight_decay': [0, 0.0001],
                    'learning_rate': [0.001, 0.0001],
                    'hidden_layers': [3,4,5,6],
                    'hidden_neurons': [128, 256, 512, 1024, 2048, 4096],
                    'num_epochs': [30,50,70],
                    'decrease_factor': [0.001, 0.0001]}
    n_bits = 4
    n_steps = 2
    
    results_df_all = random_search_soft_quantization_threshold(train_loader=train_loader,
                                                               val_loader=val_loader,
                                                               n_bits = n_bits,
                                                               n_steps = n_steps,
                                                               optimize_dict=optimize_dict,
                                                               device = device)
    results_df_all.to_csv(f'results/{dataset}/hyperparameter_tuning_{n_bits}bits.csv', index=False)