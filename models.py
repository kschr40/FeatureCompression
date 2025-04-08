import torch
import torch.nn as nn

class Combined_Model(nn.Module):
    """Combined model that consists of a start model, a quantization layer, and an end model.""" 
    def __init__(self, start_model, quantization_layer, end_model):
        """Initialize the combined model.

        Args:
            start_model (nn.Module): Start model
            quantization_layer (nn.Module): Quantization layer
            end_model (nn.Module): End model
        """
        super().__init__()
        self.start_model = start_model
        self.quantization_layer = quantization_layer
        self.end_model = end_model
        
    def forward(self, x, round_quantization = False):
        """Forward pass of the combined model.

        Args:
            x (torch.Tensor): Input tensor
            eval (bool, optional): Whether to compress the output of the quantization layer. Defaults to False.
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.start_model(x)
        x = self.quantization_layer(x, round_quantization)
        x = self.end_model(x)
        return x
    
    def get_quantization_output(self, x):
        """Get the quantization output.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.start_model(x)
        x = self.quantization_layer(x)
        return x
    
    def set_tau(self, tau):
        self.quantization_layer.tau = tau
        

class MLP_small(nn.Module):
    """Small MLP model with one layer from in_features to hidden_features, num_layers-2 layers from hidden_features to hidden_features, and one layer from hidden_features to out_features."""
    def __init__(self, in_features: int, out_features: int, hidden_features: int, num_layers: int, activation = nn.ReLU) -> None:
        """Initialize the MLP model.

        Args:
            in_features (int): Input features
            out_features (int): Output features
            hidden_features (int): Hidden features
            num_layers (int): Number of layers
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            activation(),
            *[nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                activation(),
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        """Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        return self.mlp(x)    

class MLP_small_per_feature(nn.Module):
    """Small MLP model that applies a different MLP to each feature."""
    def __init__(self, num_features, num_hidden_features):
        """Initialize the MLP model.

        Args:
            num_features (int): Number of features
            num_hidden_features (int): Number of hidden features
        """
        super().__init__()
        self.num_features = num_features

        self.feature_mlps = nn.ModuleList([nn.Sequential(nn.Linear(1,num_hidden_features),
                                                          nn.ReLU(),
                                                          nn.Linear(num_hidden_features,num_hidden_features),
                                                          nn.ReLU(),
                                                          nn.Linear(num_hidden_features,1)) for _ in range(num_features)])

    def forward(self, x):
        """Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = x.view(-1, self.num_features)
        x = torch.cat([self.feature_mlps[f](x[:,f].unsqueeze(-1)) for f in range(self.num_features)], dim = 1)
        return x        