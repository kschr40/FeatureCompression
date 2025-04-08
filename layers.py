import torch
import numpy as np
import torch.nn as nn
from typing import Union, List

class CompressionLayer(nn.Module):
    """Layer that compresses the input by thresholding the input at a given threshold.
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
    
    def forward(self, x, round_quantization = False):
        """ Forward pass of the compression layer.

        Args:
            x (torch.Tensor): Input tensor, shape [B, num_features]
            round_quantization (bool, optional): Whether to round the output. Defaults to False.

        Returns:
            torch.Tensor: Compressed tensor, shape [B, num_thresholds]
        """
        y = nn.Sigmoid()((x[:,self.a_index]-self.a) / self.tau)
        if round_quantization:
            y = torch.round(y)
        return y
    

class QuantizationLayer(nn.Module):
    """Layer that quantizes the input using learnable thresholds and a sigmoid function as approximation for step function.
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
            for f in range(self.num_features):
                x[:,f] = torch.sigmoid((x[:,f]-self.thresholds[f]) / self.tau)
        else:
            x = torch.sigmoid((x[:,:,None]-self.thresholds[None,:]) / self.tau)
            x = torch.sum(x, dim = 2)   
        if round_quantization:
            x = torch.round(x)
        return x    
    

                    





## Not in use

class FeatureSelectionLayer(nn.Module):
    """A layer that selects one of two features per output value based on a sigmoid function."""
    def __init__(self, first_index: torch.Tensor, second_index: torch.Tensor, tau: float = 1.) -> None:
        """Initializes the layer.

        Args:
            first_index (torch.Tensor): First index of comparison, shape [out_features]
            second_index (torch.Tensor): Second index of comparison, shape [out_features]
            tau (float, optional): Sharpening factor. Defaults to 1.
        """
        super().__init__()
        self.first_index = first_index
        self.second_index = second_index
        out_features = len(second_index)
        self.sigmoid_factor = nn.Parameter(torch.zeros(out_features))
        self.sigmoid = nn.Sigmoid()
        self.tau = tau

    def forward(self, x: torch.Tensor, eval = False) -> torch.Tensor:
        """Using the sigmoid_factor, we calculate how much of the first feature should be used per output value.  

        Args:
            x (torch.Tensor): input tensor, shape [B, num_features]
            eval (bool, optional): If True, sigmoid_factor is rounded (i.e. compressed). Defaults to False.

        Returns:
            torch.Tensor: Selected feature, shape [B, out_features]
        """
        self.factor = self.sigmoid(self.sigmoid_factor / self.tau)
        if eval:
            self.factor = torch.round(self.factor)
        return x[:,self.first_index] * self.factor  + x[:,self.second_index] * (1-self.factor) 
    
    def get_features(self):
        """Returns the index of the selected feature for each output value."""
        feature_index = torch.where(self.sigmoid_factor.to('cpu') >= 0, self.first_index, self.second_index)
        return feature_index