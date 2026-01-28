import torch
import numpy as np
import torch.nn as nn
from typing import Union, List

import torch.nn.functional as F

class BitwiseSoftQuantizationLayer(nn.Module):
    """Bitwise Soft Quantization Layer. Layer compresses the input by thresholding the input at a given threshold. 
    Per feature, multiple outputs in (0,1) are generated.
    """
    def __init__(self, thresholds_init: torch.Tensor, thresholds_index: torch.Tensor, tau: float = 1.):
        """Initialize the Bitwise Soft Quantization Layer.

        Args:
            thresholds_init (torch.Tensor): Initial threshold value, shape [1, num_thresholds], where num_thresholds is the total number of thresholds across all features.
            thresholds_index (torch.Tensor): thresholds_index[f] = feature index of the f-th threshold, shape [num_thresholds]
            tau (float, optional): Temperature of the sigmoid function. Defaults to 1.
        """
        super().__init__()
        self.thresholds = nn.Parameter(thresholds_init)
        self.thresholds_index = thresholds_index
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
        y = nn.Sigmoid()((x[:,self.thresholds_index]-self.thresholds) / self.tau) ## Calculation of Bitwise Soft Quantization for multiple Inputs
        if round_quantization or self.round_quantization:
            y = torch.round(y)
        return y
    
class SoftQuantizationPlusLayer(nn.Module):
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
        self.round_quantization = False
        self.tau = tau
        if isinstance(num_thresholds_per_feature, int):
            self.thresholds = nn.Parameter(torch.randn(num_features, num_thresholds_per_feature))
            self.quantized_values = nn.Parameter(torch.ones(num_features, num_thresholds_per_feature + 1))
        else:
            assert len(num_thresholds_per_feature) == num_features, "num_thresholds_per_feature must be a list of length num_features"
            self.thresholds = nn.ParameterList([nn.Parameter(torch.randn(num_thresholds)) for num_thresholds in num_thresholds_per_feature])
            self.quantized_values = nn.ParameterList([nn.Parameter(torch.ones(num_thresholds + 1)) for num_thresholds in num_thresholds_per_feature])

    def set_thresholds(self,thresholds):
        """Set the thresholds for the quantization layer.

        Args:
            thresholds (Union[torch.Tensor, List[torch.Tensor]]): Thresholds for the quantization layer. Shape [num_features, num_thresholds_per_feature] or [num_features, num_thresholds_per_feature].
        """
        if isinstance(thresholds, torch.Tensor):
            self.thresholds = nn.Parameter(thresholds)
            # thresholds = self.thresholds
            # thresholds_diffs = torch.diff(thresholds) 
            # thresholds_diffs = torch.cat([-thresholds_diffs[:,0:1], thresholds_diffs, thresholds_diffs[:,-1:]], dim=1)
            # thresholds_expanded = torch.cat([thresholds[:,0:1], thresholds], dim=1)
            # quantized_values = thresholds_expanded + thresholds_diffs/2 # Shape [num_features, num_thresholds+1] ## Rounded thresholds equal quantized values, see Equation (1)
            # self.quantized_values = nn.Parameter(quantized_values)
        else:
            assert len(thresholds) == self.num_features, "thresholds must be a list of length num_features"
            for i, threshold in enumerate(thresholds):
                self.thresholds[i] = nn.Parameter(threshold)
                # thresholds_diffs = torch.diff(threshold)
                # thresholds_diffs = torch.cat([-thresholds_diffs[0:1], thresholds_diffs, thresholds_diffs[-1:]], dim=0)
                # thresholds_expanded = torch.cat([threshold[0:1], threshold], dim=0)
                # quantized_values = thresholds_expanded + thresholds_diffs/2 # Shape [num_thresholds+1] ## Rounded thresholds equal quantized values, see Equation (1)
                # self.quantized_values[i] = nn.Parameter(quantized_values)

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
                x_bw = torch.sigmoid((x[:,f]-self.thresholds[f]) / self.tau)
                if round_quantization:
                    x_bw = torch.round(x_bw)
                x[:,f] = torch.sum(x_bw * self.quantized_values[f][None,:-1], dim = 1) + self.quantized_values[f][-1]  
        else:
            # Constant number of thresholds per feature
            x_bw = torch.sigmoid((x[:,:,None]-self.thresholds[None,:]) / self.tau) 
            if self.round_quantization:
                x_bw = torch.round(x_bw)
            x = torch.sum(x_bw, dim = 2)  #Soft Quantization = Sum of Bitwise Soft Quantization
            # x = torch.sum(x_bw * self.quantized_values.diff(dim=1)[None,:,:], dim = 2) + self.quantized_values[:,0]
        return x



class SoftQuantizationLayer(nn.Module):
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
        self.round_quantization = False
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

    def calculate_quantized_values(self):
        """Calculate the quantized values based on the thresholds.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Quantized values for each feature.
        """
        if isinstance(self.thresholds, nn.ParameterList):
            quantized_values = []
            for f in range(self.num_features):
                thresholds = self.thresholds[f]
                thresholds_diffs = torch.diff(thresholds) 
                thresholds_diffs = torch.cat([-thresholds_diffs[0:1], thresholds_diffs, thresholds_diffs[-1:]], dim=0)
                thresholds_expanded = torch.cat([thresholds[0:1], thresholds], dim=0)
                quantized_value = thresholds_expanded + thresholds_diffs/2 # Shape [num_thresholds+1] ## Rounded thresholds equal quantized values, see Equation (1)
                quantized_values.append(quantized_value)
            return quantized_values
        else:
            thresholds = self.thresholds
            thresholds_diffs = torch.diff(thresholds) 
            thresholds_diffs = torch.cat([-thresholds_diffs[:,0:1], thresholds_diffs, thresholds_diffs[:,-1:]], dim=1)
            thresholds_expanded = torch.cat([thresholds[:,0:1], thresholds], dim=1)
            quantized_values = thresholds_expanded + thresholds_diffs/2 # Shape [num_features, num_thresholds+1] ## Rounded thresholds equal quantized values, see Equation (1)
            return quantized_values            

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
                x_bw = torch.sigmoid((x[:,f]-self.thresholds[f]) / self.tau)
                if round_quantization:
                    x_bw = torch.round(x_bw)
                x[:,f] = torch.sum(x_bw, dim = 1)
        else:
            # quantized_values = self.calculate_quantized_values()
            # Constant number of thresholds per feature
            x = torch.sigmoid((x[:,:,None]-self.thresholds[None,:]) / self.tau)
            if self.round_quantization:
                x = torch.round(x)
            # x = quantized_values[:,0] + (quantized_values.diff(dim=1).unsqueeze(0) * x).sum(dim=-1)
            x = torch.sum(x, dim = 2)  #Soft Quantization = Sum of Bitwise Soft Quantization    
        # if round_quantization:
        #     x = torch.round(x)
        return x    
    
class MinMaxQuantizationLayer(nn.Module):
    def __init__(self, min_values: torch.tensor, max_values: torch.tensor, n_bits: int):
        super().__init__()
        self.register_buffer('min_values', min_values)
        self.register_buffer('max_values', max_values)
        self.num_thresholds = 2 ** n_bits - 1
        range = max_values - min_values
        self.register_buffer('range', range)
        scale = range / (self.num_thresholds)
        self.thresholds = min_values.unsqueeze(1) + scale.unsqueeze(1) * (0.5+torch.arange(0, self.num_thresholds)).unsqueeze(0)
        self.encoding_layer = ThresholdEncodingLayer(self.thresholds)
        self.decoding_layer = ThresholdDecodingLayer(self.thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the minmax quantization layer.

        Args:
            x (torch.Tensor): Input tensor to be quantized, shape [B, num_features]

        Returns:
            torch.Tensor: Quantized tensor, shape [B, num_features]
        """
        # Encoding
        indices = self.encoding_layer(x)  # Shape [B, num_features]

        # Decoding
        x_quantized = self.decoding_layer(indices)  # Shape [B, num_features]

        return x_quantized    

class QuantileQuantizationLayer(nn.Module):
    def __init__(self, thresholds:torch.Tensor):
        super().__init__()
        self.register_buffer('thresholds', thresholds)
        self.encoding_layer = ThresholdEncodingLayer(self.thresholds)
        self.decoding_layer = ThresholdDecodingLayer(self.thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the quantile quantization layer.

        Args:
            x (torch.Tensor): Input tensor to be quantized, shape [B, num_features]

        Returns:
            torch.Tensor: Quantized tensor, shape [B, num_features]
        """
        # Encoding
        indices = self.encoding_layer(x)  # Shape [B, num_features]

        # Decoding
        x_quantized = self.decoding_layer(indices)  # Shape [B, num_features]

        return x_quantized        

class ThresholdQuantizationLayer(nn.Module):
    """ Quantization Layer that performs Quantization via Thresholds."""
    def __init__(self, thresholds:torch.Tensor):
        """Initialize the hard quantization threshold layer.
        
        Args:
            thresholds (torch.Tensor): Thresholds for quantization, shape [num_features, num_thresholds]
        """
        super().__init__()
        self.register_buffer('thresholds', thresholds)
        self.encoding_layer = ThresholdEncodingLayer(thresholds)
        self.decoding_layer = ThresholdDecodingLayer(thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do hard quantization based on the thresholds.

        Args:
            x (torch.Tensor): SHape [B, num_features]

        Returns:
            torch.Tensor: Quantized tensor, shape [B, num_features]
        """
        # Encoding
        indices = self.encoding_layer(x)  # Shape [B, num_features]
        # Decoding
        x_quantized = self.decoding_layer(indices)  # Shape [B, num_features]
        return x_quantized

class ThresholdEncodingLayer(nn.Module):
    """ Quantization Layer that performs Encoding via Thresholds."""
    def __init__(self, thresholds:torch.Tensor):
        """Initialize the hard threshold encoding layer.
        
        Args:
            thresholds (torch.Tensor): Thresholds for quantization, shape [num_features, num_thresholds]
        """
        super().__init__()
        self.register_buffer('thresholds', thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do encoding based on the thresholds.

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

class ThresholdDecodingLayer(nn.Module):
    """ Decoding Layer that set quantized values in middle of the intervals between consecutive thresholds."""
    def __init__(self, thresholds:torch.Tensor):
        """Initialize the hard quantization threshold layer.
        
        Args:
            thresholds (torch.Tensor): Thresholds for quantization, shape [num_features, num_thresholds]
        """
        super().__init__()
        self.register_buffer('thresholds', thresholds)

    def forward(self, index: torch.Tensor) -> torch.Tensor:
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
        expanded_thresholds = thresholds_rounded.unsqueeze(-1).permute(0,1,2)
        expanded_index = index.permute(1,0).unsqueeze(-1).to(int)
        out = torch.gather(expanded_thresholds, 1, expanded_index).squeeze(-1).t()
        return out
    
class quant_lookup(nn.Module):
    def __init__(self, granu, n_bits, is_act=True):
        super(quant_lookup, self).__init__()
        self.n_bits = n_bits
        self.is_act = is_act ## Weights are clipped to (-1,1), activations to (0,1)
        self.granu = granu
        scale = 0.0

        if is_act:
            self.scale = nn.Parameter(torch.tensor(scale))
            self.range = 2 ** n_bits - 1
            T = torch.ones(1, 1, 2 * (granu * self.range) - 1)
            T[0, 0, (granu * self.range):] = 0
        else:
            self.scale = nn.Parameter(torch.tensor(scale))
            if n_bits == 1:
                self.range = 1
                T = torch.ones(1, 1, 2 * (granu * self.range) - 1)
                T[0, 0, (granu * self.range):] = 0
            else:
                self.range = 2 ** (n_bits - 1) - 1
                T = torch.ones(1, 1, 2 * (granu * self.range) - 1)
                T[0, 0, (granu * self.range):] = 0
        self.table = nn.Parameter(torch.zeros(self.range, granu)) ## g_i 
        self.register_buffer('T', T)

    def _update_tau(self, tau):
        self.tau = tau

    def _gen_table(self):
        if self.training:
            prob = (self.table / self.tau).softmax(1) ## p_i, shape [range, granu], sum over granu = 1
            prob = prob.view(1, 1, -1) ## shape [1, 1, range*granu]
            table_q = F.conv1d(prob, self.T, padding=prob.size(-1) - 1).unsqueeze(-1) ## shape [1, 1, range*granu, 1], equals prob.cumsum().unsqueeze(-1)
            if self.is_act:
                table_q = F.pad(table_q, [0, 0, table_q.size(2) + 1, 0])
            else:
                if self.n_bits == 1:
                    table_q = torch.cat([-torch.ones(1,1,1,1).to(prob.device), table_q * 2 - 1], 2)
                else:
                    table_q = torch.cat([-table_q.flip(2), F.pad(table_q, [0, 0, 1, 0])], 2) 
                    ## table_q.flip(2) to make negative part in ascending order
                    ## F.pad adds a zero at the beginning of second last dimension (i.e. range*granu + 1 elements), shape [1, 1, range*granu + 1, 1]
                    ## Therefore, this line effectively does: 
                    ## Puts negative part in ascending order, then adds zero, then appends positive part in ascending order

            return table_q / self.range

        else:
            index = self.table.max(1, keepdim=True)[1]
            prob = torch.zeros_like(self.table).scatter_(1, index, 1.0) ## prob is one-hot and then the same as above happens
            prob = prob.view(1, 1, -1)
            table_q = F.conv1d(prob, self.T, padding=prob.size(-1) - 1).unsqueeze(-1)
            if self.is_act:
                table_q = F.pad(table_q, [0, 0, table_q.size(2) + 1, 0])
            else:
                if self.n_bits == 1:
                    table_q = torch.cat([-torch.ones(1, 1, 1, 1).to(prob.device), table_q * 2 - 1], 2)
                else:
                    table_q = torch.cat([-table_q.flip(2), F.pad(table_q, [0, 0, 1, 0])], 2)

            return table_q / self.range

    def _lookup(self, x, table_q, scale):
        if self.training:
            grid = (x / scale).clamp(-1, 1)
            if self.is_act:
                wgt = torch.histc(grid.data, table_q.numel() // 2 + 1).float().view(1, 1, -1, 1).sqrt()
                wgt = F.pad(wgt, [0, 0, table_q.numel() // 2, 0]) + 1e-5
                table_q = table_q.data + (table_q - table_q.data) / wgt * x.numel() / (table_q.numel() // 2 + 1)
            else:
                wgt = torch.histc(grid.data, table_q.numel()).float().view(table_q.shape).sqrt() + 1e-5
                table_q = table_q.data + (table_q - table_q.data) / wgt * x.numel() / table_q.numel() ## Gradient rescaling in dependence of usage frequency (does not change forward pass)
            s = table_q.shape[2] // 2
            x_q = F.grid_sample(table_q, (F.pad(grid.data.view(1, -1, 1, 1), [1, 0]) * s).round() / s, 'nearest', 'border').view(x.shape)
            x_q = (x_q + grid - grid.data) * scale

            return x_q
        else:
            grid = (x / scale).clamp(-1, 1)
            if self.is_act:
                s = table_q.shape[2] // 2
                idx = (grid * s).round().long() + s
                x_q = table_q[0, 0, idx, 0]
            else:
                s = table_q.shape[2] - 1
                idx = ((grid + 1) / 2 * s).round().long()
                x_q = table_q[0, 0, idx, 0]
            x_q = x_q * scale

            return x_q

    def forward(self, x):
        if bool(self.scale == 0):
            if self.is_act:
                self.scale.data = (x.std() * 3).log()
            else:
                self.scale.data = (x.std() * 3).log()
        scale = self.scale.exp()

        if self.training:
            # generate lookup table
            table_q = self._gen_table()

            # lookup operation
            x_q = self._lookup(x, table_q, scale)

        else:
            table_q = self._gen_table()
            # lookup operation
            x_q = self._lookup(x, table_q, scale)

        return x_q
class quant_lookup_layer(nn.Module):
    def __init__(self, granu, n_bits, n_features, tau = 1.0):
        super(quant_lookup_layer, self).__init__()
        is_act = True
        self.quant_lookup_tables = nn.ModuleList([quant_lookup(granu, n_bits, is_act) for _ in range(n_features)])
        self.tau = tau
        for ql in self.quant_lookup_tables:
            ql._update_tau(1.0)
            ql.training = True

    def _to_device(self, device):  
        for ql in self.quant_lookup_tables:
            ql.to(device)
            
    def _update_tau(self, tau):
        self.tau = tau
        for ql in self.quant_lookup_tables:
            ql._update_tau(tau)

    def _update_training(self, training):
        for ql in self.quant_lookup_tables:
            ql.training = training
                    
    def forward(self, x):
        if hasattr(self, 'tau'):
            self._update_tau(self.tau)
        x_q = torch.tensor([], device=x.device)
        for i, quant_lookup in enumerate(self.quant_lookup_tables):
            x_q = torch.cat((x_q, quant_lookup(x[:, i:i+1])), dim=1)
        return x_q
    
import torch as t
class Quantizer(t.nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x
    

class LsqQuanTranspose(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        self.lsq_quan = LsqQuan(bit, all_positive, symmetric, per_channel)

    def init_from(self, x, *args, **kwargs):
        self.lsq_quan.init_from(x.transpose(0,1), *args, **kwargs)

    def forward(self, x):
        x = x.transpose(0,1)
        x_quant = self.lsq_quan(x)
        x_quant = x_quant.transpose(0,1)
        return x_quant        