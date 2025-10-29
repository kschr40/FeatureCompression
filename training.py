import torch
import numpy as np
import torch.nn as nn
import time

from layers import CompressionLayer, QuantizationLayer, HardQuantizationThresholdRoundingLayer, HardQuantizationLayer, HardQuantizationThresholdLayer, quant_lookup_layer, LsqQuanTranspose
from models import MultiLayerPerceptron
from layers_new import SoftQuantizationLayer, SoftQuantizationPlusLayer, BitwiseSoftQuantizationLayer, MinMaxQuantizationLayer, QuantileQuantizationLayer, ThresholdQuantizationLayer


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
def train_fp_model(architecture, min_values, max_values, quantile_thresholds,
                    train_loader, val_loader, test_loader=None,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001, add_noise=False,
                    n_bits = 8, device='cuda', save_model_path=None):
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

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)

    minmax_quantization = MinMaxQuantizationLayer(n_bits=n_bits, min_values=min_values, max_values=max_values)
    quantile_quantization = QuantileQuantizationLayer(thresholds=quantile_thresholds)

    model_PoMQ = nn.Sequential(minmax_quantization, model)
    model_PoMQ.to(device)

    model_PoQQ = nn.Sequential(quantile_quantization, model)
    model_PoQQ.to(device)

    val_loss_FP = eval_val(model=model, val_loader=val_loader, criterion=criterion, device=device)
    val_loss_PoMQ = eval_val(model=model_PoMQ, val_loader=val_loader, criterion=criterion, device=device)
    val_loss_PoQQ = eval_val(model=model_PoQQ, val_loader=val_loader, criterion=criterion, device=device)

    if test_loader is not None:
        test_loss_FP = eval_val(model=model, val_loader=test_loader, criterion=criterion, device=device)
        test_loss_PoMQ = eval_val(model=model_PoMQ, val_loader=test_loader, criterion=criterion, device=device)
        test_loss_PoQQ = eval_val(model=model_PoQQ, val_loader=test_loader, criterion=criterion, device=device)
    else:
        test_loss_FP = test_loss_PoMQ = test_loss_PoQQ = None

    return val_loss_FP, val_loss_PoMQ, val_loss_PoQQ, test_loss_FP, test_loss_PoMQ, test_loss_PoQQ, loss_training_last_epoch, elapsed_time


def train_pre_model(architecture, min_values, max_values, quantile_thresholds,
                    train_loader, val_loader, test_loader=None,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001, add_noise=False,
                    n_bits = 8, device='cuda', load_model_path=None):
    minmax_quantization = MinMaxQuantizationLayer(n_bits=n_bits, min_values=min_values, max_values=max_values)
    quantile_quantization = QuantileQuantizationLayer(thresholds=quantile_thresholds)

    mlp_MQ = MultiLayerPerceptron(architecture)
    mlp_QQ = MultiLayerPerceptron(architecture)
    if load_model_path is not None:
        mlp_MQ.load_state_dict(torch.load(load_model_path,weights_only=True))
        mlp_QQ.load_state_dict(torch.load(load_model_path,weights_only=True))

    model_PrMQ = nn.Sequential(minmax_quantization, mlp_MQ)
    model_PrMQ.to(device)

    model_PrQQ = nn.Sequential(quantile_quantization, mlp_QQ)
    model_PrQQ.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_PrMQ.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    best_val_loss_PrMQ, loss_training_last_epoch_mm = train_model(model_PrMQ, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion, has_quantization_layer=False,
                train_quantization_layer=False, print_result=False,
                add_noise=add_noise, device=device)
    end = time.perf_counter()
    elapsed_time = end - start

    val_loss_PrMQ = eval_val(model=model_PrMQ, val_loader=val_loader, criterion=criterion, device=device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_PrQQ.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_loss_PrQQ, loss_training_last_epoch_q = train_model(model_PrQQ, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion, has_quantization_layer=False,
                train_quantization_layer=False, print_result=False,
                add_noise=add_noise, device=device)

    val_loss_PrQQ = eval_val(model=model_PrQQ, val_loader=val_loader, criterion=criterion, device=device)

    if test_loader is not None:
        test_loss_PrMQ = eval_val(model=model_PrMQ, val_loader=test_loader, criterion=criterion, device=device)
        test_loss_PrQQ = eval_val(model=model_PrQQ, val_loader=test_loader, criterion=criterion, device=device)
    else:
        test_loss_PrMQ = test_loss_PrQQ = None

    return val_loss_PrMQ, val_loss_PrQQ, test_loss_PrMQ, test_loss_PrQQ, loss_training_last_epoch_mm, loss_training_last_epoch_q, elapsed_time


def train_SQ(architecture, min_values, max_values, quantile_thresholds,
                    train_loader, val_loader, test_loader=None,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001, add_noise=False,
                    n_bits = 8, decrease_factor = 0.001, device='cuda', load_model_path=None):
    num_features = quantile_thresholds.shape[0]
    num_thresholds_per_feature = quantile_thresholds.shape[1]
    model_SQ_quantizer = SoftQuantizationLayer(num_features=quantile_thresholds.shape[0], 
                                     num_thresholds_per_feature=quantile_thresholds.shape[1],
                                     tau=1)

    model_SQ_quantizer.set_thresholds(quantile_thresholds)
    mlp = MultiLayerPerceptron(architecture)

    if load_model_path is not None:
        mlp.load_state_dict(torch.load(load_model_path,weights_only=True))

    model_SQ = nn.Sequential(model_SQ_quantizer, mlp)
    model_SQ.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_SQ.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    best_val_loss, loss_training_last_epoch = train_model(model_SQ, num_epochs=num_epochs,
                train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion, has_quantization_layer=True,
                train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
                add_noise=add_noise, device=device)
    end = time.perf_counter()
    elapsed_time = end - start
    
    hard_quantization_layer = ThresholdQuantizationLayer(thresholds=model_SQ_quantizer.thresholds)
    model_SQ_inf = nn.Sequential(hard_quantization_layer, mlp)

    val_loss_SQ_train = eval_val(model=model_SQ, val_loader=val_loader, criterion=criterion, device=device)
    val_loss_SQ_inf = eval_val(model=model_SQ_inf, val_loader=val_loader, criterion=criterion, device=device)

    if test_loader is not None:
        test_loss_SQ_train = eval_val(model=model_SQ, val_loader=test_loader, criterion=criterion, device=device)
        test_loss_SQ_inf = eval_val(model=model_SQ_inf, val_loader=test_loader, criterion=criterion, device=device)
    else:
        test_loss_SQ_train = test_loss_SQ_inf = None

    return val_loss_SQ_train, val_loss_SQ_inf, test_loss_SQ_train, test_loss_SQ_inf, loss_training_last_epoch, elapsed_time


def train_SQplus(architecture, min_values, max_values, quantile_thresholds,
                    train_loader, val_loader, test_loader=None,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001, add_noise=False,
                    n_bits = 8, decrease_factor = 0.001, device='cuda', load_model_path=None):
    num_features = quantile_thresholds.shape[0]
    num_thresholds_per_feature = quantile_thresholds.shape[1]

    quantization_model = SoftQuantizationPlusLayer(num_features=quantile_thresholds.shape[0], 
                                                    num_thresholds_per_feature=quantile_thresholds.shape[1],
                                                    tau=1)
    quantization_model.set_thresholds(quantile_thresholds)

    mlp = MultiLayerPerceptron(architecture)

    if load_model_path is not None:
        mlp.load_state_dict(torch.load(load_model_path,weights_only=True))

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
    
    val_loss_soft_mlp = eval_val(model=model_soft_mlp, val_loader=val_loader, criterion=criterion, device=device)
    quantization_model.round_quantization = True
    val_loss_soft_hard_mlp = eval_val(model=model_soft_mlp, val_loader=val_loader, criterion=criterion, device=device)

    if test_loader is not None:
        # restore round flag for test evaluation as appropriate
        quantization_model.round_quantization = False
        test_loss_soft_mlp = eval_val(model=model_soft_mlp, val_loader=test_loader, criterion=criterion, device=device)
        quantization_model.round_quantization = True
        test_loss_soft_hard_mlp = eval_val(model=model_soft_mlp, val_loader=test_loader, criterion=criterion, device=device)
    else:
        test_loss_soft_mlp = test_loss_soft_hard_mlp = None

    return val_loss_soft_mlp, val_loss_soft_hard_mlp, test_loss_soft_mlp, test_loss_soft_hard_mlp, loss_training_last_epoch, elapsed_time


def train_BwSQ(architecture, min_values, max_values, quantile_thresholds,
                    train_loader, val_loader, test_loader=None,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001, add_noise=False,
                    n_bits = 8, decrease_factor = 0.001, device='cuda'):
    num_features = quantile_thresholds.shape[0]
    num_thresholds_per_feature = quantile_thresholds.shape[1]

    comp_model = CompressionLayer(a_init=quantile_thresholds.flatten(),
                                  a_index=torch.repeat_interleave(torch.arange(num_features), num_thresholds_per_feature),
                                  tau=1)
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

    if test_loader is not None:
        comp_model.set_round_quantization(False)
        test_loss_soft_thr_mlp = eval_val(model=model_soft_thr_mlp, val_loader=test_loader, criterion=criterion, device=device)
        comp_model.set_round_quantization(True)
        test_loss_soft_hard_thr_mlp = eval_val(model=model_soft_thr_mlp, val_loader=test_loader, criterion=criterion, device=device)
    else:
        test_loss_soft_thr_mlp = test_loss_soft_hard_thr_mlp = None

    return val_loss_soft_thr_mlp, val_loss_soft_hard_thr_mlp, test_loss_soft_thr_mlp, test_loss_soft_hard_thr_mlp, loss_training_last_epoch, elapsed_time


def train_BwMQ_BwQQ(architecture, minmax_thresholds, quantile_thresholds,
                    train_loader, val_loader, test_loader=None,
                    num_epochs=100, learning_rate=0.001, weight_decay=0.0001, add_noise=False,
                    n_bits = 8, decrease_factor = 0.001, device='cuda'):
    num_features = quantile_thresholds.shape[0]
    num_thresholds_per_feature = quantile_thresholds.shape[1]

    comp_thr_model = CompressionLayer(a_init=quantile_thresholds.flatten(),
                                  a_index=torch.repeat_interleave(torch.arange(num_features), num_thresholds_per_feature),
                                  tau=1)
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

    if test_loader is not None:
        # For quantile-based
        comp_q = CompressionLayer(a_init=quantile_thresholds.flatten(),
                                  a_index=torch.repeat_interleave(torch.arange(num_features), num_thresholds_per_feature),
                                  tau=1)
        comp_q.set_round_quantization(True)
        model_q = nn.Sequential(comp_q, MultiLayerPerceptron(architecture))
        model_q.to(device)
        test_loss_hard_bitwise_quantile_mlp = eval_val(model=model_q, val_loader=test_loader, criterion=criterion, device=device)

        # For minmax-based
        comp_m = CompressionLayer(a_init=minmax_thresholds.flatten(),
                                  a_index=torch.repeat_interleave(torch.arange(num_features), num_thresholds_per_feature),
                                  tau=1)
        comp_m.set_round_quantization(True)
        model_m = nn.Sequential(comp_m, MultiLayerPerceptron(architecture))
        model_m.to(device)
        test_loss_hard_bitwise_minmax_mlp = eval_val(model=model_m, val_loader=test_loader, criterion=criterion, device=device)
    else:
        test_loss_hard_bitwise_minmax_mlp = test_loss_hard_bitwise_quantile_mlp = None

    return val_loss_hard_bitwise_minmax_mlp, val_loss_hard_bitwise_quantile_mlp, test_loss_hard_bitwise_minmax_mlp, test_loss_hard_bitwise_quantile_mlp, loss_training_last_epoch_mm, loss_training_last_epoch_q, elapsed_time


def train_llt(architecture, min_values, max_values, quantile_thresholds,
              train_loader, val_loader, test_loader=None,
              num_epochs=100, learning_rate=0.001, weight_decay=0.0001, add_noise=False,
              n_bits = 8, decrease_factor = 0.001, device='cuda', load_model_path=None):
    num_features = quantile_thresholds.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp = MultiLayerPerceptron(architecture)

    if load_model_path is not None:
        mlp.load_state_dict(torch.load(load_model_path,weights_only=True))

    # When we have 2 bits we can create 4 thresholds (|) - if I have granularity 2 each "region" has 2 additional thresholds (i)
    # ---i----i---|---i----i---|---i----i---|---i----i---|---i----i---
    # TODO: this does not exactly match the size of the table?
    model_llt = nn.Sequential(quant_lookup_layer(granu=10, n_bits=n_bits,n_features=num_features),mlp)

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

    if test_loader is not None:
        model_llt[0]._update_training(True)
        test_loss_llt_training = eval_val(model=model_llt, val_loader=test_loader, criterion=criterion, device=device)
        model_llt[0]._update_training(False)
        test_loss_llt = eval_val(model=model_llt, val_loader=test_loader, criterion=criterion, device=device)
    else:
        test_loss_llt_training = test_loss_llt = None

    return val_loss_llt, val_loss_llt_training, test_loss_llt, test_loss_llt_training, loss_training_last_epoch, elapsed_time


def train_lsq(architecture, min_values, max_values, quantile_thresholds,
              train_loader, val_loader, test_loader=None,
              num_epochs=100, learning_rate=0.001, weight_decay=0.0001, add_noise=False,
              n_bits = 8, decrease_factor = 0.001, device='cuda', load_model_path=None):
    num_features = quantile_thresholds.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mlp = MultiLayerPerceptron(architecture)
    if load_model_path is not None:
        mlp.load_state_dict(torch.load(load_model_path,weights_only=True))

    model_lsq = nn.Sequential(LsqQuanTranspose(bit=n_bits,all_positive=False,symmetric=False,per_channel=True),
                      mlp)
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

    if test_loader is not None:
        test_loss_lsq = eval_val(model=model_lsq, val_loader=test_loader, criterion=criterion, device=device)
    else:
        test_loss_lsq = None
 
    return val_loss_lsq, test_loss_lsq, loss_training_last_epoch, elapsed_time

# def train_fp_model(architecture, min_values, max_values, quantile_thresholds,
#                     train_loader, val_loader,
#                     num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
#                     n_bits = 8, device='cuda', save_model_path=None):
#     model = MultiLayerPerceptron(architecture)

#     model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
#     start = time.perf_counter()
#     best_val_loss, loss_training_last_epoch = train_model(model, num_epochs=num_epochs,
#                 train_loader=train_loader, val_loader=val_loader,
#                 optimizer=optimizer, criterion=criterion, has_quantization_layer=False,
#                 train_quantization_layer=False, print_result=False,
#                 add_noise=add_noise, device=device)
#     end = time.perf_counter()
#     elapsed_time = end - start

#     if save_model_path is not None:
#         torch.save(model.state_dict(), save_model_path)

#     minmax_quantization = MinMaxQuantizationLayer(n_bits=n_bits, min_values=min_values, max_values=max_values)
#     quantile_quantization = QuantileQuantizationLayer(thresholds=quantile_thresholds)

#     model_PoMQ = nn.Sequential(minmax_quantization, model)
#     model_PoMQ.to(device)

#     model_PoQQ = nn.Sequential(quantile_quantization, model)
#     model_PoQQ.to(device)

#     val_loss_FP = eval_val(model=model, val_loader=val_loader, criterion=criterion, device=device)
#     val_loss_PoMQ = eval_val(model=model_PoMQ, val_loader=val_loader, criterion=criterion, device=device)
#     val_loss_PoQQ = eval_val(model=model_PoQQ, val_loader=val_loader, criterion=criterion, device=device)
#     return val_loss_FP, val_loss_PoMQ, val_loss_PoQQ, loss_training_last_epoch, elapsed_time


# def train_pre_model(architecture, min_values, max_values, quantile_thresholds,
#                     train_loader, val_loader, 
#                     num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
#                     n_bits = 8, device='cuda', load_model_path=None):
#     minmax_quantization = MinMaxQuantizationLayer(n_bits=n_bits, min_values=min_values, max_values=max_values)
#     quantile_quantization = QuantileQuantizationLayer(thresholds=quantile_thresholds)

#     mlp_MQ = MultiLayerPerceptron(architecture)
#     mlp_QQ = MultiLayerPerceptron(architecture)
#     if load_model_path is not None:
#         mlp_MQ.load_state_dict(torch.load(load_model_path,weights_only=True))
#         mlp_QQ.load_state_dict(torch.load(load_model_path,weights_only=True))

#     model_PrMQ = nn.Sequential(minmax_quantization, mlp_MQ)
#     model_PrMQ.to(device)

#     model_PrQQ = nn.Sequential(quantile_quantization, mlp_QQ)
#     model_PrQQ.to(device)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model_PrMQ.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     start = time.perf_counter()
#     best_val_loss_PrMQ, loss_training_last_epoch_mm = train_model(model_PrMQ, num_epochs=num_epochs,
#                 train_loader=train_loader, val_loader=val_loader,
#                 optimizer=optimizer, criterion=criterion, has_quantization_layer=False,
#                 train_quantization_layer=False, print_result=False,
#                 add_noise=add_noise, device=device)
#     end = time.perf_counter()
#     elapsed_time = end - start

#     val_loss_PrMQ = eval_val(model=model_PrMQ, val_loader=val_loader, criterion=criterion, device=device)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model_PrQQ.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     best_val_loss_PrQQ, loss_training_last_epoch_q = train_model(model_PrQQ, num_epochs=num_epochs,
#                 train_loader=train_loader, val_loader=val_loader,
#                 optimizer=optimizer, criterion=criterion, has_quantization_layer=False,
#                 train_quantization_layer=False, print_result=False,
#                 add_noise=add_noise, device=device)

#     val_loss_PrQQ = eval_val(model=model_PrQQ, val_loader=val_loader, criterion=criterion, device=device)
#     return val_loss_PrMQ, val_loss_PrQQ, loss_training_last_epoch_mm, loss_training_last_epoch_q, elapsed_time

# def train_SQ(architecture, min_values, max_values, quantile_thresholds,
#                     train_loader, val_loader,
#                     num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
#                     n_bits = 8, decrease_factor = 0.001, device='cuda', load_model_path=None):
#     num_features = quantile_thresholds.shape[0]
#     num_thresholds_per_feature = quantile_thresholds.shape[1]
#     model_SQ_quantizer = SoftQuantizationLayer(num_features=quantile_thresholds.shape[0], 
#                                      num_thresholds_per_feature=quantile_thresholds.shape[1],
#                                      tau=1)

#     model_SQ_quantizer.set_thresholds(quantile_thresholds)
#     mlp = MultiLayerPerceptron(architecture)

#     if load_model_path is not None:
#         mlp.load_state_dict(torch.load(load_model_path,weights_only=True))

#     model_SQ = nn.Sequential(model_SQ_quantizer, mlp)
#     model_SQ.to(device)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model_SQ.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     start = time.perf_counter()
#     best_val_loss, loss_training_last_epoch = train_model(model_SQ, num_epochs=num_epochs,
#                 train_loader=train_loader, val_loader=val_loader,
#                 optimizer=optimizer, criterion=criterion, has_quantization_layer=True,
#                 train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
#                 add_noise=add_noise, device=device)
#     end = time.perf_counter()
#     elapsed_time = end - start
    
#     hard_quantization_layer = ThresholdQuantizationLayer(thresholds=model_SQ_quantizer.thresholds)
#     model_SQ_inf = nn.Sequential(hard_quantization_layer, mlp)

#     val_loss_SQ_train = eval_val(model=model_SQ, val_loader=val_loader, criterion=criterion, device=device)
#     val_loss_SQ_inf = eval_val(model=model_SQ_inf, val_loader=val_loader, criterion=criterion, device=device)

#     return val_loss_SQ_train, val_loss_SQ_inf, loss_training_last_epoch, elapsed_time

# def train_SQplus(architecture, min_values, max_values, quantile_thresholds,
#                     train_loader, val_loader,
#                     num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
#                     n_bits = 8, decrease_factor = 0.001, device='cuda', load_model_path=None):
#     num_features = quantile_thresholds.shape[0]
#     num_thresholds_per_feature = quantile_thresholds.shape[1]

#     quantization_model = SoftQuantizationPlusLayer(num_features=quantile_thresholds.shape[0], 
#                                                     num_thresholds_per_feature=quantile_thresholds.shape[1],
#                                                     tau=1)
#     quantization_model.set_thresholds(quantile_thresholds)

#     mlp = MultiLayerPerceptron(architecture)

#     if load_model_path is not None:
#         mlp.load_state_dict(torch.load(load_model_path,weights_only=True))

#     model_soft_mlp = nn.Sequential(quantization_model, mlp)
#     model_soft_mlp.to(device)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model_soft_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     start = time.perf_counter()
#     best_val_loss, loss_training_last_epoch = train_model(model_soft_mlp, num_epochs=num_epochs,
#                 train_loader=train_loader, val_loader=val_loader,
#                 optimizer=optimizer, criterion=criterion, has_quantization_layer=True,
#                 train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
#                 add_noise=add_noise, device=device)
#     end = time.perf_counter()
#     elapsed_time = end - start
    
#     val_loss_soft_mlp = eval_val(model=model_soft_mlp, val_loader=val_loader, criterion=criterion, device=device)
#     quantization_model.round_quantization = True
#     val_loss_soft_hard_mlp = eval_val(model=model_soft_mlp, val_loader=val_loader, criterion=criterion, device=device)

#     return val_loss_soft_mlp, val_loss_soft_hard_mlp, loss_training_last_epoch, elapsed_time

# def train_BwSQ(architecture, min_values, max_values, quantile_thresholds,
#                     train_loader, val_loader,
#                     num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
#                     n_bits = 8, decrease_factor = 0.001, device='cuda'):
#     num_features = quantile_thresholds.shape[0]
#     num_thresholds_per_feature = quantile_thresholds.shape[1]

#     comp_model = CompressionLayer(a_init=quantile_thresholds.flatten(),
#                                   a_index=torch.repeat_interleave(torch.arange(num_features), num_thresholds_per_feature),
#                                   tau=1)
#     architecture[0] = num_features * num_thresholds_per_feature
#     mlp = MultiLayerPerceptron(architecture)



#     model_soft_thr_mlp = nn.Sequential(comp_model, mlp)
#     model_soft_thr_mlp.to(device)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model_soft_thr_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     start = time.perf_counter()
#     best_val_loss, loss_training_last_epoch = train_model(model_soft_thr_mlp, num_epochs=num_epochs,
#                 train_loader=train_loader, val_loader=val_loader,
#                 optimizer=optimizer, criterion=criterion,has_quantization_layer=True,
#                 train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
#                 add_noise=add_noise, device=device)
#     end = time.perf_counter()
#     elapsed_time = end - start
    
#     val_loss_soft_thr_mlp = eval_val(model=model_soft_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)
#     comp_model.set_round_quantization(True)   
#     val_loss_soft_hard_thr_mlp = eval_val(model=model_soft_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)
#     return val_loss_soft_thr_mlp, val_loss_soft_hard_thr_mlp, loss_training_last_epoch, elapsed_time

# def train_BwMQ_BwQQ(architecture, minmax_thresholds, quantile_thresholds,
#                     train_loader, val_loader,
#                     num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
#                     n_bits = 8, decrease_factor = 0.001, device='cuda'):
#     num_features = quantile_thresholds.shape[0]
#     num_thresholds_per_feature = quantile_thresholds.shape[1]

#     # quantization_model = HardQuantizationLayer(n_bits=n_bits, min_values=min_values, max_values=max_values)

#     comp_thr_model = CompressionLayer(a_init=quantile_thresholds.flatten(),
#                                   a_index=torch.repeat_interleave(torch.arange(num_features), num_thresholds_per_feature),
#                                   tau=1)
#     comp_thr_model.set_round_quantization(True)
#     architecture[0] = num_features * num_thresholds_per_feature
#     mlp = MultiLayerPerceptron(architecture)

#     model_hard_thr_mlp = nn.Sequential(comp_thr_model, mlp)
#     model_hard_thr_mlp.to(device)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model_hard_thr_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     start = time.perf_counter()
#     best_val_loss, loss_training_last_epoch_mm = train_model(model_hard_thr_mlp, num_epochs=num_epochs,
#                 train_loader=train_loader, val_loader=val_loader,
#                 optimizer=optimizer, criterion=criterion,has_quantization_layer=True,
#                 train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
#                 add_noise=add_noise, device=device)
#     end = time.perf_counter()
#     elapsed_time = end - start

#     val_loss_hard_bitwise_quantile_mlp = eval_val(model=model_hard_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)
    
#     comp_thr_model = CompressionLayer(a_init = minmax_thresholds.flatten(),
#                                   a_index = torch.repeat_interleave(torch.arange(num_features), num_thresholds_per_feature),
#                                   tau = 1)
#     comp_thr_model.set_round_quantization(True)
#     architecture[0] = num_features * num_thresholds_per_feature
#     mlp = MultiLayerPerceptron(architecture)

#     model_hard_thr_mlp = nn.Sequential(comp_thr_model, mlp)
#     model_hard_thr_mlp.to(device)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model_hard_thr_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     best_val_loss, loss_training_last_epoch_q = train_model(model_hard_thr_mlp, num_epochs=num_epochs,
#                 train_loader=train_loader, val_loader=val_loader,
#                 optimizer=optimizer, criterion=criterion,has_quantization_layer=True,
#                 train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
#                 add_noise=add_noise, device=device)
    
#     val_loss_hard_bitwise_minmax_mlp = eval_val(model=model_hard_thr_mlp, val_loader=val_loader, criterion=criterion, device=device)

#     return val_loss_hard_bitwise_minmax_mlp, val_loss_hard_bitwise_quantile_mlp, loss_training_last_epoch_mm, loss_training_last_epoch_q, elapsed_time

# def train_llt(architecture, min_values, max_values, quantile_thresholds,
#               train_loader, val_loader,
#               num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
#               n_bits = 8, decrease_factor = 0.001, device='cuda', load_model_path=None):
#     num_features = quantile_thresholds.shape[0]
#     from models import MultiLayerPerceptron
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     mlp = MultiLayerPerceptron(architecture)

#     if load_model_path is not None:
#         mlp.load_state_dict(torch.load(load_model_path,weights_only=True))

#     model_llt = nn.Sequential(quant_lookup_layer(granu=9, n_bits=n_bits,n_features=num_features),mlp)

#     model_llt.to(device)

#     criterion = nn.MSELoss()
#     model_llt[0]._update_tau(1.0)
#     model_llt[0]._update_training(True)
#     optimizer = torch.optim.Adam(model_llt.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     start = time.perf_counter()
#     best_val_loss, loss_training_last_epoch = train_model(model_llt, num_epochs=num_epochs,
#                 train_loader=train_loader, val_loader=val_loader,
#                 optimizer=optimizer, criterion=criterion,has_quantization_layer=True,
#                 train_quantization_layer=True, print_result=False, decrease_factor=decrease_factor,
#                 add_noise=add_noise, device=device)
#     end = time.perf_counter()
#     elapsed_time = end - start
    
#     model_llt[0]._update_training(True)
#     val_loss_llt_training = eval_val(model=model_llt, val_loader=val_loader, criterion=criterion, device=device)
#     model_llt[0]._update_training(False)
#     val_loss_llt = eval_val(model=model_llt, val_loader=val_loader, criterion=criterion, device=device)
#     return val_loss_llt, val_loss_llt_training, loss_training_last_epoch, elapsed_time

# def train_lsq(architecture, min_values, max_values, quantile_thresholds,
#               train_loader, val_loader,
#               num_epochs=100, learning_rate=0.001, weight_decay=0.0001,add_noise=False,
#               n_bits = 8, decrease_factor = 0.001, device='cuda', load_model_path=None):
#     num_features = quantile_thresholds.shape[0]
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     mlp = MultiLayerPerceptron(architecture)
#     if load_model_path is not None:
#         mlp.load_state_dict(torch.load(load_model_path,weights_only=True))

#     model_lsq = nn.Sequential(LsqQuanTranspose(bit=n_bits,all_positive=False,symmetric=False,per_channel=True),
#                       mlp)
#     X_train = torch.vstack([x for x, y in train_loader])
#     model_lsq[0].init_from(X_train)

#     model_lsq.to(device)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model_lsq.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     start = time.perf_counter()
#     best_val_loss, loss_training_last_epoch = train_model(model_lsq, num_epochs=num_epochs,
#                 train_loader=train_loader, val_loader=val_loader,
#                 optimizer=optimizer, criterion=criterion,has_quantization_layer=False,
#                 train_quantization_layer=False, print_result=False, decrease_factor=decrease_factor,
#                 add_noise=add_noise, device=device)
#     end = time.perf_counter()
#     elapsed_time = end - start
    
#     val_loss_lsq = eval_val(model=model_lsq, val_loader=val_loader, criterion=criterion, device=device)
 
#     return val_loss_lsq, loss_training_last_epoch, elapsed_time