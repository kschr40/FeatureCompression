import torch
import numpy as np
import torch.nn as nn
import time

from layers import CompressionLayer, QuantizationLayer, HardQuantizationThresholdRoundingLayer, HardQuantizationLayer, HardQuantizationThresholdLayer, quant_lookup_layer, LsqQuanTranspose
from models import MultiLayerPerceptron


def eval_val(model, val_loader, criterion = nn.MSELoss(), device = 'cuda'):
    """Evaluate performance of the model on the validation set.

    Args:
        model: model to use 
        val_dataloader: dataloader of validation set
        eval (bool, optional): If True, all outputs of sigmoids will be rounded (i.e. compressed). Defaults to False.
        device (str, optional): device. Defaults to 'cuda'.

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        # model.eval()
        losses = []
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)    
            loss = criterion(output, y)
            losses.append(loss.item())
        return np.mean(losses)

def train_model(model: nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, 
                num_epochs: int, 
                has_quantization_layer: bool = True, 
                decrease_factor: float = 0.001, 
                train_quantization_layer: bool = True, 
                device: str = 'cuda', 
                print_result: bool = True,
                add_noise = False):
    """
    Trains a model

    Args:
        model (nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): dataloader of training set
        test_loader (torch.utils.data.DataLoader): dataloader of validation set
        optimizer (torch.optim.Optimizer): optimizer to use
        criterion (nn.Module): criterion to use
        num_epochs (int): number of epochs to train
        has_quantization_layer (bool): whether the model has a quantization layer
        decrease_factor (float): factor to decrease the tau of the quantization layer
        train_quantization_layer (bool): whether to train the quantization layer
        device (str): device to use
        print_result (bool): whether to print the result
    """
    model.train()
    if not train_quantization_layer and has_quantization_layer:
        for param in model.quantization_layer.parameters():
            param.requires_grad = False
    factor = decrease_factor ** (1/num_epochs)
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
        if has_quantization_layer:
            model[0].tau = max(model[0].tau * factor, 0.0001)
            # model.set_tau(max(model.quantization_layer.tau * factor, 0.001))
        val_loss = eval_val(model, val_loader, criterion = criterion, device = device)
        if epoch % (num_epochs//10) == 0 and print_result:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(losses)}, Val Loss: {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_model = model.state_dict()
    return best_val_loss, train_loss


## Training specific models

def train_mlp_model(architecture, min_values, max_values, thresholds,
                    train_loader, val_loader,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
                    n_bits = 8, device='cuda'):
    model = MultiLayerPerceptron(architecture)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    start = time.perf_counter()
    best_val_loss, loss_training_last_epoch = train_model(model, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion, has_quantization_layer=False,
                train_quantization_layer=False, print_result=False,
                add_noise=add_noise, device=device)
    end = time.perf_counter()
    elapsed_time = end - start
    
    quantization_model = HardQuantizationLayer(n_bits=n_bits, min_values=min_values, max_values=max_values)
    quantization_thr_model = HardQuantizationThresholdRoundingLayer(thresholds=thresholds)

    model_hard_post_mlp = nn.Sequential(quantization_model, model)
    model_hard_post_mlp.to(device)

    model_hard_thr_post_mlp = nn.Sequential(quantization_thr_model, model)
    model_hard_thr_post_mlp.to(device)

    val_loss_mlp = eval_val(model=model, val_loader=val_loader, criterion=criterion, device=device)
    val_loss_hard_post_mlp = eval_val(model=model_hard_post_mlp, val_loader=val_loader, criterion=criterion, device=device)
    val_loss_hard_thr_post_mlp = eval_val(model=model_hard_thr_post_mlp, val_loader=val_loader, criterion=criterion, device=device)
    return val_loss_mlp, val_loss_hard_post_mlp, val_loss_hard_thr_post_mlp, loss_training_last_epoch, elapsed_time
    


def train_mlp_pre_model(architecture, min_values, max_values, thresholds,
                    train_loader, val_loader, 
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
                    n_bits = 8, device='cuda'):
    quantization_model = HardQuantizationLayer(n_bits=n_bits, min_values=min_values, max_values=max_values)
    quantization_thr_model = HardQuantizationThresholdLayer(thresholds=thresholds)
    mlp = MultiLayerPerceptron(architecture)
    mlp_thr = MultiLayerPerceptron(architecture)

    model_hard_pre_mlp = nn.Sequential(quantization_model, mlp)
    model_hard_pre_mlp.to(device)

    model_hard_pre_thr_mlp = nn.Sequential(quantization_thr_model, mlp_thr)
    model_hard_pre_thr_mlp.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_hard_pre_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    best_val_loss_hard_pre_mlp, loss_training_last_epoch_mm = train_model(model_hard_pre_mlp, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion, has_quantization_layer=False,
                train_quantization_layer=False, print_result=False,
                add_noise=add_noise, device=device)
    end = time.perf_counter()
    elapsed_time = end - start
    
    val_loss_hard_pre_mlp = eval_val(model=model_hard_pre_mlp, val_loader=val_loader, criterion=criterion, device=device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_hard_pre_thr_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_loss_hard_pre_thr_mlp, loss_training_last_epoch_q = train_model(model_hard_pre_thr_mlp, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion, has_quantization_layer=False,
                train_quantization_layer=False, print_result=False,
                add_noise=add_noise, device=device)
    
    val_loss_hard_pre_thr_mlp = eval_val(model=model_hard_pre_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)
    return val_loss_hard_pre_mlp, val_loss_hard_pre_thr_mlp, loss_training_last_epoch_mm, loss_training_last_epoch_q, elapsed_time

def train_soft_mlp(architecture, min_values, max_values, thresholds,
                    train_loader, val_loader,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
                    n_bits = 8, decrease_factor = 0.001, device='cuda'):
    num_features = thresholds.shape[0]
    num_thresholds_per_feature = thresholds.shape[1]

    quantization_model = QuantizationLayer(num_features=thresholds.shape[0], 
                                           num_thresholds_per_feature=thresholds.shape[1],
                                           tau=1)
    quantization_model.set_thresholds(thresholds)
    mlp = MultiLayerPerceptron(architecture)

    model_soft_mlp = nn.Sequential(quantization_model, mlp)
    model_soft_mlp.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_soft_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    best_val_loss, loss_training_last_epoch = train_model(model_soft_mlp, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion, has_quantization_layer=True,
                train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
                add_noise=add_noise, device=device)
    end = time.perf_counter()
    elapsed_time = end - start
    
    quantization_thr_model = HardQuantizationThresholdLayer(thresholds=quantization_model.thresholds)
    model_soft_hard_mlp = nn.Sequential(quantization_thr_model, mlp)

    val_loss_soft_mlp = eval_val(model=model_soft_mlp, val_loader=val_loader, criterion=criterion, device=device)
    val_loss_soft_hard_mlp = eval_val(model=model_soft_hard_mlp, val_loader=val_loader, criterion=criterion, device=device)

    return val_loss_soft_mlp, val_loss_soft_hard_mlp, loss_training_last_epoch, elapsed_time

def train_soft_comp_mlp(architecture, min_values, max_values, thresholds,
                    train_loader, val_loader,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
                    n_bits = 8, decrease_factor = 0.001, device='cuda'):
    num_features = thresholds.shape[0]
    num_thresholds_per_feature = thresholds.shape[1]
    
   
    comp_model = CompressionLayer(a_init = thresholds.flatten(), 
                                  a_index = torch.repeat_interleave(torch.arange(num_features),num_thresholds_per_feature), 
                                  tau = 1)
    architecture[0] = num_features * num_thresholds_per_feature
    mlp = MultiLayerPerceptron(architecture)

    model_soft_thr_mlp = nn.Sequential(comp_model, mlp)
    model_soft_thr_mlp.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_soft_thr_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    best_val_loss, loss_training_last_epoch = train_model(model_soft_thr_mlp, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion,has_quantization_layer=True,
                train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
                add_noise=add_noise, device=device)
    end = time.perf_counter()
    elapsed_time = end - start
    
    val_loss_soft_thr_mlp = eval_val(model=model_soft_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)
    comp_model.set_round_quantization(True)   
    val_loss_soft_hard_thr_mlp = eval_val(model=model_soft_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)
    return val_loss_soft_thr_mlp, val_loss_soft_hard_thr_mlp, loss_training_last_epoch, elapsed_time

def train_hard_comp_mlp(architecture, minmax_thresholds, thresholds,
                    train_loader, val_loader,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
                    n_bits = 8, decrease_factor = 0.001, device='cuda'):
    num_features = thresholds.shape[0]
    num_thresholds_per_feature = thresholds.shape[1]

    # quantization_model = HardQuantizationLayer(n_bits=n_bits, min_values=min_values, max_values=max_values)

    comp_thr_model = CompressionLayer(a_init = thresholds.flatten(), 
                                  a_index = torch.repeat_interleave(torch.arange(num_features),num_thresholds_per_feature), 
                                  tau = 1)
    comp_thr_model.set_round_quantization(True)
    architecture[0] = num_features * num_thresholds_per_feature
    mlp = MultiLayerPerceptron(architecture)

    model_hard_thr_mlp = nn.Sequential(comp_thr_model, mlp)
    model_hard_thr_mlp.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_hard_thr_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    best_val_loss, loss_training_last_epoch_mm = train_model(model_hard_thr_mlp, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion,has_quantization_layer=True,
                train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
                add_noise=add_noise, device=device)
    end = time.perf_counter()
    elapsed_time = end - start

    val_loss_hard_bitwise_quantile_mlp = eval_val(model=model_hard_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)
    
    comp_thr_model = CompressionLayer(a_init = minmax_thresholds.flatten(),
                                  a_index = torch.repeat_interleave(torch.arange(num_features), num_thresholds_per_feature),
                                  tau = 1)
    comp_thr_model.set_round_quantization(True)
    architecture[0] = num_features * num_thresholds_per_feature
    mlp = MultiLayerPerceptron(architecture)

    model_hard_thr_mlp = nn.Sequential(comp_thr_model, mlp)
    model_hard_thr_mlp.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_hard_thr_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_loss, loss_training_last_epoch_q = train_model(model_hard_thr_mlp, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion,has_quantization_layer=True,
                train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
                add_noise=add_noise, device=device)
    
    val_loss_hard_bitwise_minmax_mlp = eval_val(model=model_hard_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)

    return val_loss_hard_bitwise_minmax_mlp, val_loss_hard_bitwise_quantile_mlp, loss_training_last_epoch_mm, loss_training_last_epoch_q, elapsed_time

def train_llt(architecture, min_values, max_values, thresholds,
              train_loader, val_loader,
              num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
              n_bits = 8, decrease_factor = 0.001, device='cuda'):
    num_features = thresholds.shape[0]
    from models import MultiLayerPerceptron
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_llt = nn.Sequential(quant_lookup_layer(granu=9, n_bits=n_bits,n_features=num_features),MultiLayerPerceptron(architecture))

    model_llt.to(device)

    criterion = nn.MSELoss()
    model_llt[0]._update_tau(1.0)
    model_llt[0]._update_training(True)
    optimizer = torch.optim.Adam(model_llt.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    best_val_loss, loss_training_last_epoch = train_model(model_llt, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion,has_quantization_layer=True,
                train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
                add_noise=add_noise, device=device)
    end = time.perf_counter()
    elapsed_time = end - start
    
    model_llt[0]._update_training(True)
    val_loss_llt_training = eval_val(model=model_llt, val_loader=val_loader, criterion=criterion, device=device)
    model_llt[0]._update_training(False)
    val_loss_llt = eval_val(model=model_llt, val_loader=val_loader, criterion=criterion, device=device)
    return val_loss_llt, val_loss_llt_training, loss_training_last_epoch, elapsed_time

def train_lsq(architecture, min_values, max_values, thresholds,
              train_loader, val_loader,
              num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
              n_bits = 8, decrease_factor = 0.001, device='cuda'):
    num_features = thresholds.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_lsq = nn.Sequential(LsqQuanTranspose(bit=n_bits,all_positive=False,symmetric=False,per_channel=True),
                      MultiLayerPerceptron(architecture))
    X_train = torch.vstack([x for x, y in train_loader])
    model_lsq[0].init_from(X_train)

    model_lsq.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_lsq.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    best_val_loss, loss_training_last_epoch = train_model(model_lsq, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion,has_quantization_layer=False,
                train_quantization_layer=False, print_result=False, decrease_factor=decrease_factor,
                add_noise=add_noise, device=device)
    end = time.perf_counter()
    elapsed_time = end - start
    
    val_loss_lsq = eval_val(model=model_lsq, val_loader=val_loader, criterion=criterion, device=device)
 
    return val_loss_lsq, loss_training_last_epoch, elapsed_time