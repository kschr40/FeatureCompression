import torch
from torch.utils.data import DataLoader, TensorDataset

from layers import CompressionLayer, QuantizationLayer, FeatureSelectionLayer, HardQuantizationLayer, HardQuantizationThresholdLayer
from models import MultiLayerPerceptron
from datasets import get_dataloader
from notebooks.training_utils import train_model, eval_val, eval_quantization

def get_model(input_size, output_size, hidden_size, num_layers, quantization_type, 
              quantization_params, device):
    """
    Get a model with the specified parameters.

    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        hidden_size (int): Size of the hidden layers.
        num_layers (int): Number of hidden layers.
        quantization_type (str): Type of quantization to use.
        quantization_params (dict): Parameters for the quantization layer.
        feature_selection_params (dict): Parameters for the feature selection layer.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        nn.Module: The constructed model.
    """
    model = MultiLayerPerceptron(input_size, output_size, hidden_size, num_layers)
    
    if quantization_type == 'quantization':
        model.quantization_layer = QuantizationLayer(**quantization_params).to(device)
    elif quantization_type == 'compression':
        model.quantization_layer = CompressionLayer(**quantization_params).to(device)
    elif quantization_type == 'hard_quantization':
        model.quantization_layer = HardQuantizationLayer(**quantization_params).to(device)
    elif quantization_type == 'hard_quantization_threshold':
        model.quantization_layer = HardQuantizationThresholdLayer(**quantization_params).to(device)
        
    return model