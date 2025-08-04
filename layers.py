import torch
import numpy as np
import torch.nn as nn
from typing import Union, List

class CompressionLayer(nn.Module):
    """Bitwise Soft Quantization Layer. Layer compresses the input by thresholding the input at a given threshold. Per feature, multiple outputs in (0,1) are generated.
    """
    def __init__(self, a_init: torch.Tensor, a_index: torch.Tensor, tau: float = 1.):
        """Initialize the compression layer.

        Args:
            a_init (torch.Tensor): Initial threshold value, shape [1, num_thresholds]
            a_index (torch.Tensor): a_index[f] = feature index of the f-th threshold, shape [num_thresholds]
            tau (float, optional): Temperature of the sigmoid function. Defaults to 1.
        """
        super().__init__()
        self.a = nn.Parameter(a_init)
        self.a_index = a_index
        self.tau = tau
        self.round_quantization = False

    def set_round_quantization(self, round_quantization: bool):
        """Set the round quantization flag.

        Args:
            round_quantization (bool): Whether to round the output.
        """
        self.round_quantization = round_quantization    
    
    def forward(self, x, round_quantization = False):
        """ Forward pass of the compression layer.

        Args:
            x (torch.Tensor): Input tensor, shape [B, num_features]
            round_quantization (bool, optional): Whether to round the output. Defaults to False.

        Returns:
            torch.Tensor: Compressed tensor, shape [B, num_thresholds]
        """
        y = nn.Sigmoid()((x[:,self.a_index]-self.a) / self.tau) ## Calculation of Bitwise Soft Quantization for multiple Inputs
        if round_quantization or self.round_quantization:
            y = torch.round(y)
        return y
    

class QuantizationLayer(nn.Module):
    """Soft Quantization Layer. Layer that quantizes the input using learnable thresholds and a sigmoid function as approximation for step function. Per features, a single output in (0, n_thresholds_per_feature) are generated.
    """
    def __init__(self, num_features: int, num_thresholds_per_feature: Union[int, List[int]], tau: float = 1.):
        """Initialize the quantization layer.

        Args:
            num_features (int): Number of features in the input
            num_thresholds_per_feature (Union[int, List[int]]): Number of thresholds per feature. Either a single integer or a list of integers of length num_features.
            tau (float, optional): Temperature of the sigmoid function. Defaults to 1.
        """
        super().__init__()
        self.num_features = num_features
        self.num_thresholds_per_feature = num_thresholds_per_feature
        self.tau = tau
        if isinstance(num_thresholds_per_feature, int):
            self.thresholds = nn.Parameter(torch.randn(num_features, num_thresholds_per_feature))
        else:
            assert len(num_thresholds_per_feature) == num_features, "num_thresholds_per_feature must be a list of length num_features"
            self.thresholds = nn.ParameterList([nn.Parameter(torch.randn(num_thresholds)) for num_thresholds in num_thresholds_per_feature])

    def set_thresholds(self,thresholds):
        """Set the thresholds for the quantization layer.

        Args:
            thresholds (Union[torch.Tensor, List[torch.Tensor]]): Thresholds for the quantization layer. Shape [num_features, num_thresholds_per_feature] or [num_features, num_thresholds_per_feature].
        """
        if isinstance(thresholds, torch.Tensor):
            self.thresholds = nn.Parameter(thresholds)
        else:
            assert len(thresholds) == self.num_features, "thresholds must be a list of length num_features"
            for i, threshold in enumerate(thresholds):
                self.thresholds[i] = nn.Parameter(threshold)

    def forward(self, x, round_quantization = False):
        """Forward pass of the quantization layer.

        Args:
            x (torch.Tensor): Input tensor, shape [B, num_features]
            round_quantization (bool, optional): Whether to round the output. Defaults to False.

        Returns:
            torch.Tensor: Quantized tensor, shape [B, num_features]
        """
        if isinstance(self.thresholds, nn.ParameterList):
            # Varying number of thresholds per feature
            for f in range(self.num_features):
                x[:,f] = torch.sum(torch.sigmoid((x[:,f]-self.thresholds[f]) / self.tau), dim = 1)
        else:
            # Constant number of thresholds per feature
            x = torch.sigmoid((x[:,:,None]-self.thresholds[None,:]) / self.tau)
            x = torch.sum(x, dim = 2)  #Soft Quantization = Sum of Bitwise Soft Quantization    
        if round_quantization:
            x = torch.round(x)
        return x    
    
                    
class HardQuantizationLayer(nn.Module):
    """Minmax Quantization Layer. A layer that quantizes inputs into n_bits bits using minmax thresholding."""
    def __init__(self, n_bits: int, min_values: torch.tensor, max_values: torch.tensor):
        """Initialize the hard quantization layer.
        
        Args:
            n_bits (int): Number of bits to quantize into
            min_values (torch.tensor): Minimum values for each feature
            max_values (torch.tensor): Maximum values for each feature
        """
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2**n_bits
        self.register_buffer('min_values', min_values)
        self.register_buffer('max_values', max_values)
        self.register_buffer('range', max_values - min_values)
        
    def forward(self, x: torch.Tensor, do_quantization = True) -> torch.Tensor:
        """Forward pass of the hard quantization layer.
        
        Args:
            x (torch.Tensor): Input tensor to be quantized
            round_quantization (bool): Unused parameter kept for compatibility
            
        Returns:
            torch.Tensor: Quantized tensor with values in [0,1]
        """
        if do_quantization:
            # Scale x to [0,1] based on min/max values
            x = (x - self.min_values) / self.range
            x = torch.clamp(x, 0, 1)
            
            # Quantize to n_bits
            x = torch.round(x * (self.n_levels - 1)) / (self.n_levels - 1) ## Encoded values, see Appendix A for details.
            
            # Scale back to original range
            x = x * self.range + self.min_values ## Decoded values, i.e. quantized values, according to Equation (1). Details in Appendix A.
        
        return x

class HardQuantizationThresholdLayer(nn.Module):
    """ Quantization Layer that performs Encoding via Thresholds and Identity for Decoding."""
    def __init__(self, thresholds:torch.Tensor):
        """Initialize the hard quantization threshold layer.
        
        Args:
            thresholds (torch.Tensor): Thresholds for quantization, shape [num_features, num_thresholds]
        """
        super().__init__()
        self.register_buffer('thresholds', thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do hard quantization based on the thresholds.

        Args:
            x (torch.Tensor): SHape [B, num_features]

        Returns:
            torch.Tensor: Quantized tensor, shape [B, num_features]
        """
        x = x.unsqueeze(-1)  # Add a dimension for broadcasting
        # Compare x with thresholds
        x = (x > self.thresholds.unsqueeze(0)).float() ## Shape [B, num_features, num_thresholds]   
        # Sum along the threshold dimension to get the final quantized values (also equals encoded values, See Section Encoding Function via Thresholds)
        x = torch.sum(x, dim=-1)  # Shape [B, num_features]
        return x

class HardQuantizationThresholdRoundingLayer(nn.Module):
    """ Quantization Layer that performs Encoding via Thresholds and set quantized values in middle of the intervals between consecutive thresholds."""
    def __init__(self, thresholds:torch.Tensor):
        """Initialize the hard quantization threshold layer.
        
        Args:
            thresholds (torch.Tensor): Thresholds for quantization, shape [num_features, num_thresholds]
        """
        super().__init__()
        self.register_buffer('thresholds', thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do hard quantization based on the thresholds.

        Args:
            x (torch.Tensor): SHape [B, num_features]

        Returns:
            torch.Tensor: Quantized tensor, shape [B, num_features]
        """
        # Auxiliary thresholds are added and quantized values are computed
        thresholds = self.thresholds
        thresholds_diffs = torch.diff(thresholds) 
        thresholds_diffs = torch.cat([-thresholds_diffs[:,0:1], thresholds_diffs, thresholds_diffs[:,-1:]], dim=1)
        thresholds = torch.cat([thresholds[:,0:1], thresholds], dim=1)
        thresholds_rounded = thresholds + thresholds_diffs/2 # Shape [num_features, num_thresholds+1] ## Rounded thresholds equal quantized values, see Equation (1)

        # Calculate the correct quantized values - only calculation of indexes is left
        x = x.unsqueeze(-1)  # Add a dimension for broadcasting
        # Compare x with thresholds
        x = (x > self.thresholds.unsqueeze(0)).float() ## Shape [B, num_features, num_thresholds]   
        # Sum along the threshold dimension to get the final quantized values
        index = torch.sum(x, dim=-1)  # Shape [B, num_features]
        expanded_thresholds = thresholds_rounded.unsqueeze(-1).permute(0,1,2)
        expanded_index = index.permute(1,0).unsqueeze(-1).to(int)
        out = torch.gather(expanded_thresholds, 1, expanded_index).squeeze(-1).t()
        return out