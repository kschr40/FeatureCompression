import torch
import torch.nn as nn
import numpy as np

def eval_val(model, val_dataloader, criterion = nn.MSELoss(), eval= False, device = 'cuda'):
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
        model.eval()
        losses = []
        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            if eval:
                output = model(x, round_quantization=True)    
            else:    
                output = model(x) 
                if isinstance(output, tuple):
                    output = output[1]
            loss = criterion(output, y)
            losses.append(loss.item())
        return np.mean(losses)

def train_model(model: nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                test_loader: torch.utils.data.DataLoader, 
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
        if has_quantization_layer:
            model[0].tau = max(model[0].tau * factor, 0.0001)
            # model.set_tau(max(model.quantization_layer.tau * factor, 0.001))
        val_loss = eval_val(model, test_loader, criterion = criterion, device = device)
        if epoch % (num_epochs//10) == 0 and print_result:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(losses)}, Val Loss: {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_model = model.state_dict()
    return best_val_loss


def eval_quantization(model, val_dataloader, device = 'cuda'):
    """Calculates, how far the model is from giving compressed hidden states.

    Args:
        model: important: model.forward() must return hidden_state, output
        val_dataloader: dataloader of validation set
        device (str, optional): device. Defaults to 'cuda'.

    Returns:
        : mean of the absolute difference between hidden state and rounded (i.e. compressed) hidden state
    """
    with torch.no_grad():
        model.eval()
        losses = []
        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            hidden_state = model.get_quantization_output(x)  
            hidden_state_compressed = torch.round(hidden_state) 
            loss = torch.mean(torch.abs(hidden_state - hidden_state_compressed))
            losses.append(loss.item())
        return np.mean(losses)