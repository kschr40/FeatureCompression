from datasets import get_min_max_values, get_quantization_thresholds, load_data, get_minmax_thresholds
import torch
import pandas as pd
import random
from tqdm import tqdm
from training import eval_val
import os

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import time

from models import MultiLayerPerceptron
from layers import BitwiseSoftQuantizationLayer


def create_tau_schedule(min_val=0.001, max_val = 1, name = 'linear', steps = 10):
    """Create a schedule for the temperature hyperparameter tau. 

    Args:
        min_val (float, optional): Minimum value of tau. Defaults to 0.001.
        max_val (int, optional): Maximum value of tau. Defaults to 1.
        name (str, optional): Name of the schedule. Defaults to 'linear'.
        steps (int, optional): Number of steps in the schedule. Defaults to 10.

    Raises:
        ValueError: If an unknown schedule name is provided.

    Returns:
        np.ndarray: Array containing the tau schedule.
    """
    if name == 'linear':
        tau_schedule = np.linspace(max_val, min_val, steps)
    elif name == 'exponential':
        tau_schedule = np.logspace(np.log10(max_val), np.log10(min_val), steps)
    elif name == 'warmup':
        tau_schedule = []
        half_steps = steps // 2
        tau_schedule += list(np.linspace(max_val, max_val, half_steps, endpoint=False))
        tau_schedule += list(np.linspace(max_val, min_val, steps - half_steps, endpoint = True))
    elif name == 'cosine':
        tau_schedule = [min_val + (max_val - min_val) * (1 + np.cos(np.pi * i / (steps - 1))) / 2 for i in range(steps)]
    elif name == 'cyclical':
        tau_schedule = []
        third_steps = steps // 3
        # First third: max to min
        tau_schedule += list(np.linspace(max_val, min_val, third_steps, endpoint=False))
        # Second third: min to max
        tau_schedule += list(np.linspace(min_val, max_val, third_steps, endpoint=False))
        # Third third: max to min
        tau_schedule += list(np.linspace(max_val, min_val, steps - 2 * third_steps, endpoint=True))        
    else:
        raise ValueError(f"Unknown tau schedule name: {name}")
    return tau_schedule

def train_model(model: nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, 
                num_epochs: int, 
                has_quantization_layer: bool = True, 
                train_quantization_layer: bool = True, 
                device: str = 'cuda', 
                print_result: bool = True,
                add_noise = False, 
                tau_schedule = None):
    """
    Trains a model. Slight changes compared to training.train_model to accomodate tau scheduling.

    Args:
        model (nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): dataloader of training set
        val_loader (torch.utils.data.DataLoader): dataloader of validation set
        optimizer (torch.optim.Optimizer): optimizer to use
        criterion (nn.Module): criterion to use
        num_epochs (int): number of epochs to train
        has_quantization_layer (bool): whether the model has a quantization layer
        train_quantization_layer (bool): whether to train the quantization layer
        device (str): device to use
        print_result (bool): whether to print the result
    """
    model.train()
    if not train_quantization_layer and has_quantization_layer:
        for param in model.quantization_layer.parameters():
            param.requires_grad = False
    best_val_loss = float('inf')
    train_loss = 0
    for epoch in range(num_epochs):
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            if add_noise:
                x += torch.randn_like(x) / 25
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)
        if tau_schedule is not None:
            model[0].tau = tau_schedule[min(epoch+1, num_epochs-1)]
            # model.set_tau(max(model.quantization_layer.tau * factor, 0.001))
        val_loss = eval_val(model, val_loader, criterion = criterion, device = device)
        if epoch % (num_epochs//10) == 0 and print_result:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(losses)}, Val Loss: {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_model = model.state_dict()
    return best_val_loss, train_loss

def train_BwSQ(architecture, quantile_thresholds,
                    train_loader, val_loader, test_loader=None,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001, add_noise=False,
                    decrease_factor = 0.001, device='cuda', dropout = 0.0, tau_schedule=None):
    ''' Train a Bitwise Soft Quantization model followed by an MLP, slight changes to training.train_BwSQ to accomodate tau scheduling.'''
    num_features = quantile_thresholds.shape[0]
    num_thresholds_per_feature = quantile_thresholds.shape[1]

    comp_model = BitwiseSoftQuantizationLayer(thresholds_init=quantile_thresholds.flatten(),
                                  thresholds_index=torch.repeat_interleave(torch.arange(num_features), num_thresholds_per_feature),
                                  tau=1)
    architecture[0] = num_features * num_thresholds_per_feature
    mlp = MultiLayerPerceptron(architecture, dropout=dropout)



    model_soft_thr_mlp = nn.Sequential(comp_model, mlp)
    model_soft_thr_mlp.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_soft_thr_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    best_val_loss, loss_training_last_epoch = train_model(model_soft_thr_mlp, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion,has_quantization_layer=True,
                train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
                add_noise=add_noise, device=device, tau_schedule=tau_schedule)
    end = time.perf_counter()
    elapsed_time = end - start
    model_soft_thr_mlp.eval()
    val_loss_soft_thr_mlp = eval_val(model=model_soft_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)
    comp_model.set_round_quantization(True)   
    model_soft_thr_mlp.eval()
    val_loss_soft_hard_thr_mlp = eval_val(model=model_soft_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)

    if test_loader is not None:
        comp_model.set_round_quantization(False)
        test_loss_soft_thr_mlp = eval_val(model=model_soft_thr_mlp, val_loader=test_loader, criterion=criterion, device=device)
        comp_model.set_round_quantization(True)
        test_loss_soft_hard_thr_mlp = eval_val(model=model_soft_thr_mlp, val_loader=test_loader, criterion=criterion, device=device)
    else:
        test_loss_soft_thr_mlp = test_loss_soft_hard_thr_mlp = None

    return val_loss_soft_thr_mlp, val_loss_soft_hard_thr_mlp, test_loss_soft_thr_mlp, test_loss_soft_hard_thr_mlp, loss_training_last_epoch, elapsed_time



def test_different_schedules_cv(X_tensor : torch.tensor, y_tensor : torch.tensor, result_folder, dataset,
                            k_folds = 4, n_bits =8,  device = 'cpu', ):
    
    num_features = X_tensor.shape[1]

    X_cv_array, X_test_array, y_cv_array, y_test_array = train_test_split(X_tensor.numpy(), y_tensor.numpy(), test_size=0.1, random_state=42)

    # Define hyperparameters

    # 13,3,2,0,0.001,6,512,50,0.001,0.4 for Wine dataset 2 bits (best hyperparameter setting)
    # 17,3,7,0,0.0001,6,8192,50,0.0001,0.2, for Superconduct dataset 7 bits (best hyperparameter setting)

    weight_decay =  0
    learning_rate = 0.0001
    hidden_layers = 6
    hidden_neurons = 8192
    num_epochs = 50
    decrease_factor = 0.0001
    dropout_rate = 0.2

    min_vals = [0.01,0.001,0.0001]
    max_vals = [1]

    schedules = ['linear', 'exponential', 'warmup', 'cosine', 'cyclical']
    result_df = pd.DataFrame(index=range(k_folds * len(schedules) * len(min_vals) * len(max_vals)))


    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    kfold_split = kfold.split(X_cv_array)
    splits = list(kfold_split)
    for f in tqdm(range(len(splits))):
        train_idx, val_idx = splits[f]
        
        SEED = 42 + 2*f + (17*k_folds + f) * k_folds

        # Set seeds for Python, NumPy, and PyTorch
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        # Preprocess data
        X_train_tensor = X_cv_array[train_idx]
        y_train_tensor = y_cv_array[train_idx]
        X_val_tensor = X_cv_array[val_idx]
        y_val_tensor = y_cv_array[val_idx]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_array = scaler_X.fit_transform(X_train_tensor)
        y_train_array = scaler_y.fit_transform(y_train_tensor)
        X_val_array = scaler_X.transform(X_val_tensor)
        y_val_array = scaler_y.transform(y_val_tensor)
        X_test_array_scaled = scaler_X.transform(X_test_array)
        y_test_array_scaled = scaler_y.transform(y_test_array)

        X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_array, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_array, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_array_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_array_scaled, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


        quantile_thresholds = get_quantization_thresholds(train_loader, n_bits)

        architecture = [num_features] + [hidden_neurons] * hidden_layers + [1]
        for n_schedule, schedule in tqdm(enumerate(schedules), total=len(schedules)):
            for n_min_val, min_val in enumerate(min_vals):
                for n_max_val, max_val in enumerate(max_vals):
                    current_row = f * len(schedules) * len(min_vals) * len(max_vals) + n_schedule * len(min_vals) * len(max_vals) + n_min_val * len(max_vals) + n_max_val
                    result_df.at[current_row, 'n_bits'] = n_bits
                    result_df.at[current_row, 'schedule'] = schedule
                    result_df.at[current_row, 'fold'] = f
                    result_df.at[current_row, 'min_val'] = min_val
                    result_df.at[current_row, 'max_val'] = max_val

                    tau_schedule = create_tau_schedule(min_val=min_val, 
                                                    max_val = max_val, 
                                                    name = schedule, 
                                                    steps = num_epochs)

                # Calculate losses for bitwise soft quantization model
                    val_loss_BwSQ_train, val_loss_BwSQ_inf, test_loss_BwSQ_train, test_loss_BwSQ_inf, train_loss_BwSQ, time_BwSQ = train_BwSQ(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, architecture=architecture, quantile_thresholds=quantile_thresholds,
                        num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout_rate,
                        n_bits=n_bits, decrease_factor=decrease_factor, device=device, tau_schedule=tau_schedule)
                    result_df.at[current_row, 'val_loss_Bw-SQ'] = val_loss_BwSQ_inf
                    result_df.at[current_row, 'val_loss_Bw-SQ_train'] = val_loss_BwSQ_train
                    result_df.at[current_row, 'test_loss_Bw-SQ'] = test_loss_BwSQ_inf
                    result_df.at[current_row, 'test_loss_Bw-SQ_train'] = test_loss_BwSQ_train
                    result_df.at[current_row, 'train_loss_Bw-SQ'] = train_loss_BwSQ
                    result_df.at[current_row, 'time_Bw-SQ'] = time_BwSQ

        folder = Path(f'{result_folder}/{dataset}')
        folder.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(f'{result_folder}/{dataset}/{dataset}_bwsq_tau_{n_bits}bits_{f+1}iterations_new.csv', index=False)
        if f > 0:
            os.remove(f'{result_folder}/{dataset}/{dataset}_bwsq_tau_{n_bits}bits_{f}iterations_new.csv')
    return result_df

if __name__ == "__main__":
    # dataset = 'wine_quality'
    dataset = 'superconduct'
    scratch = 'data'
    n_bits = 7
    k_folds = 10
    result_folder = 'results'

    X_tensor, y_tensor = load_data(dataset, scratch, False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results_df_all = test_different_schedules_cv(X_tensor=X_tensor, y_tensor=y_tensor,
                                                    result_folder=result_folder,
                                                    dataset=dataset,
                                                    n_bits=n_bits,
                                                    device=device,
                                                    k_folds=k_folds)
    # print(f"Finished random search for {n_steps} steps with {n_bits} bits on dataset {dataset}")
