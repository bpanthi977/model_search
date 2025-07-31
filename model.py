from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import re

import torch
import torch.nn as nn

from config import TrainConfig, ModelConfig, parse_hidden_layers
from dataset import Dataset


leaky_relu_slope = 1e-2

def activation_function(act: str):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(leaky_relu_slope)
    else:
        raise ValueError(f"Invalid value for activation {act}")

def init_weights(m: nn.Module, config: ModelConfig):
    if isinstance(m, nn.Linear):
        init = config.init

        if init == 'u' or init == 'udd':
            assert(len(config.init_param) == 2)
            nn.init.uniform_(m.weight, config.init_param[0], config.init_param[1])

            if init == 'udd':
                row_sums = torch.sum(m.weight, dim=1)

                # Create a diagonal matrix (rectangular) that matches the size of weight
                out_features, in_features = m.weight.shape
                padded_diag = torch.zeros((out_features, in_features), dtype=m.weight.dtype, device=m.weight.device)
                diag_size = min(out_features, in_features)
                padded_diag[:diag_size, :diag_size] = torch.diag(row_sums[:diag_size])

                m.weight.data += padded_diag

        elif init == 'ku':
            if config.activation == 'leaky_relu':
                nn.init.kaiming_uniform_(m.weight, a=leaky_relu_slope, nonlinearity='leaky_relu', mode='fan_in')
            else:
                nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu', mode='fan_in')

        elif init == 'xu':
            if config.activation == 'leaky_relu':
                gain = nn.init.calculate_gain(config.activation, leaky_relu_slope)
            else:
                gain = nn.init.calculate_gain(config.activation)

            nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init == 'default':
            pass
        else:
            assert False, "Impossible initialization parameter. Should have been checked in ModelConfig."

@dataclass
class Normalization:
    X_mu: Optional[torch.Tensor]
    X_std: Optional[torch.Tensor]
    Y_mu: Optional[torch.Tensor]
    Y_std: Optional[torch.Tensor]

    def to(self, device: torch.DeviceObjType):
        if self.X_mu != None:
            self.X_mu = self.X_mu.to(device)
        if self.X_std != None:
            self.X_std = self.X_std.to(device)
        if self.Y_mu != None:
            self.Y_mu = self.Y_mu.to(device)
        if self.Y_std != None:
            self.Y_std = self.Y_std.to(device)

    def X(self):
        if self.X_mu != None and self.X_std != None:
            return (True, self.X_mu, self.X_std)
        else:
            return (False, torch.tensor(1), torch.tensor(1))

    def Y(self):
        if self.Y_mu != None and self.Y_std != None:
            return (True, self.Y_mu, self.Y_std)
        else:
            return (False, torch.tensor(1), torch.tensor(1))

class NALUi1(nn.Module):
    def __init__(self, input_dim, output_dim, dtype: torch.dtype):
        super().__init__()
        self.w_hat = nn.Parameter(torch.empty(input_dim, output_dim, dtype=dtype))
        self.m_hat = nn.Parameter(torch.empty(input_dim, output_dim, dtype=dtype))
        self.g = nn.Parameter(torch.empty(output_dim, dtype=dtype))

        nn.init.xavier_uniform_(self.w_hat)
        nn.init.xavier_uniform_(self.m_hat)
        nn.init.constant_(self.g, 0.0)

    def forward(self, x):
        W = torch.tanh(self.w_hat) * torch.sigmoid(self.m_hat)
        a = x @ W
        m = torch.exp(torch.clamp((torch.log(torch.clamp(torch.abs(x), min=1e-7)) @ W), max=20.0))

        # Sign tracking
        W_flat = torch.abs(W.flatten())
        x_tiled = x.repeat(1, W.shape[1])
        x_reshaped = x_tiled.view(-1, W.shape[0] * W.shape[1])
        sgn = torch.sign(x_reshaped) * W_flat + (1 - W_flat)
        sgn = sgn.view(-1, W.shape[1], W.shape[0])
        ms = torch.prod(sgn, dim=2)

        g_val = torch.sigmoid(self.g)
        out = g_val * a + (1 - g_val) * m * torch.clamp(ms, -1, 1)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.w_hat.shape[1]}, out_features={self.w_hat.shape[0]})"

class NALUi2(nn.Module):
    def __init__(self, input_dim, output_dim, dtype: torch.dtype):
        super().__init__()
        self.w_hat1 = nn.Parameter(torch.empty(input_dim, output_dim, dtype=dtype))
        self.m_hat1 = nn.Parameter(torch.empty(input_dim, output_dim, dtype=dtype))
        self.w_hat2 = nn.Parameter(torch.empty(input_dim, output_dim, dtype=dtype))
        self.m_hat2 = nn.Parameter(torch.empty(input_dim, output_dim, dtype=dtype))
        self.g = nn.Parameter(torch.empty(output_dim, dtype=dtype))

        nn.init.xavier_uniform_(self.w_hat1)
        nn.init.xavier_uniform_(self.m_hat1)
        nn.init.xavier_uniform_(self.w_hat2)
        nn.init.xavier_uniform_(self.m_hat2)
        nn.init.constant_(self.g, 0.0)

    def forward(self, x):
        W1 = torch.tanh(self.w_hat1) * torch.sigmoid(self.m_hat1)
        W2 = torch.tanh(self.w_hat2) * torch.sigmoid(self.m_hat2)
        a = x @ W1
        m = torch.exp(torch.clamp((torch.log(torch.clamp(torch.abs(x), min=1e-7)) @ W2), max=20.0))

        # Sign tracking
        W2_flat = torch.abs(W2.flatten())
        x_tiled = x.repeat(1, W1.shape[1])
        x_reshaped = x_tiled.view(-1, W1.shape[0] * W1.shape[1])
        sgn = torch.sign(x_reshaped) * W2_flat + (1 - W2_flat)
        sgn = sgn.view(-1, W1.shape[1], W1.shape[0])
        ms = torch.prod(sgn, dim=2)

        g_val = torch.sigmoid(self.g)
        out = g_val * a + (1 - g_val) * m * torch.clamp(ms, -1, 1)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.w_hat1.shape[1]}, out_features={self.w_hat1.shape[0]})"

class MULT0(nn.Module):
    def __init__(self, input_dim, output_dim, dtype: torch.dtype):
        super().__init__()
        self.w_hat = nn.Parameter(torch.empty(input_dim, output_dim, dtype=dtype))
        self.m_hat = nn.Parameter(torch.empty(input_dim, output_dim, dtype=dtype))

        nn.init.xavier_uniform_(self.w_hat)
        nn.init.xavier_uniform_(self.m_hat)

    def forward(self, x):
        W = torch.tanh(self.w_hat) * torch.sigmoid(self.m_hat)
        m = torch.exp(torch.clamp((torch.log(torch.clamp(torch.abs(x), min=1e-7)) @ W), max=20.0))

        return m

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.w_hat.shape[1]}, out_features={self.w_hat.shape[0]})"

class NALU(nn.Module):
    """
    Neural Arithmetic Logic Units.
    Layer designed to learn arithmetic: +, -, *, / exactly.
    """
    def __init__(self, input_dim, output_dim, dtype: torch.dtype):
        super().__init__()
        self.W_hat = nn.Parameter(torch.empty(output_dim, input_dim, dtype=dtype))
        self.M_hat = nn.Parameter(torch.empty(output_dim, input_dim, dtype=dtype))
        self.G = nn.Parameter(torch.empty(output_dim, input_dim, dtype=dtype))
        self.reset_parameters()

    @torch.jit.ignore
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_hat)
        nn.init.kaiming_uniform_(self.M_hat)
        nn.init.kaiming_uniform_(self.G)

    def forward(self, x):
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        a = torch.matmul(x, W.T)

        log_x = torch.log(torch.abs(x) + 1e-7)
        m = torch.exp(torch.matmul(log_x, W.T))
        g = torch.sigmoid(torch.matmul(x, self.G.T))

        return g * a + (1 - g) * m

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.W_hat.shape[1]}, out_features={self.W_hat.shape[0]})"

class MLP(nn.Module):
    def __init__(self, device, input_dim, output_dim, config: ModelConfig, normalize: Normalization):
        super().__init__()
        self.device = device
        normalize.to(device)
        self.normalizeX = normalize.X()
        self.normalizeY = normalize.Y()

        layers = []
        fan_in = input_dim
        last_layer = 'linear'

        @torch.jit.ignore
        def add_layer(layer_type, layer_dim):
            if layer_type == 'linear':
                layers.append(nn.Linear(fan_in, layer_dim, dtype=torch.float64, bias=config.bias))
            elif layer_type == 'NALU':
                layers.append(NALU(fan_in, layer_dim, dtype=torch.float64))
            elif layer_type == 'NALUi1':
                layers.append(NALUi1(fan_in, layer_dim, dtype=torch.float64))
            elif layer_type == 'NALUi2':
                layers.append(NALUi1(fan_in, layer_dim, dtype=torch.float64))
            elif layer_type == 'MULT0':
                layers.append(MULT0(fan_in, layer_dim, dtype=torch.float64))
            else:
                raise ValueError(f'[BUG] layer_type {layer_type} not implemented.')

        for (i, (layer_type, layer_dim, layer_activation)) in enumerate(parse_hidden_layers(config.hidden_layers)):
            if layer_dim == 0:
                last_layer = layer_type
                break
            add_layer(layer_type, layer_dim)
            fan_in = layer_dim

            if config.batchnorm:
                layers.append(nn.BatchNorm1d(layer_dim, dtype=torch.float64))

            if not layer_activation and not layer_activation == '':
                layers.append(activation_function(config.activation))
            elif layer_activation == 'I':
                pass # No activation
            else:
                raise ValueError(f"[BUG] layer_activation {layer_activation} not implemented.")
            if i < len(config.dropout):
                layers.append(nn.Dropout(config.dropout[i]))

        add_layer(last_layer, output_dim)
        self.model = nn.Sequential(*layers)
        self.model.apply(lambda m: init_weights(m, config))
        self.to(device)

    def normalize(self, t: torch.Tensor, normalize: Tuple[bool, torch.Tensor, torch.Tensor]):
        if normalize[0]:
            t = (t - normalize[1]) / normalize[2]

        return t

    def denormalize(self, t: torch.Tensor, normalize: Tuple[bool, torch.Tensor, torch.Tensor]):
        if normalize[0]:
            t = t * normalize[2] + normalize[1]

        return t

    def forward(self, x):
        y = self.model.forward(self.normalize(x, self.normalizeX))
        # Model is trained on normalized X and Y
        # but we need to return un-normalized Y to the user
        y = self.denormalize(y, self.normalizeY)

        return y

    def get_device(self):
        return self.device

    def set_device(self, device):
        self.normalizeX[1].to(device)
        self.normalizeY[1].to(device)
        self.to(device)
        self.device = device

def create_model(config: TrainConfig, dataset: Dataset):
    return MLP(
        torch.device(config.device),
        dataset.input_dim(),
        dataset.output_dim(),
        config.model,
        Normalization(
            dataset.trainX.mean(dim=0) if config.model.normalizeX else None,
            dataset.trainX.std(dim=0)  if config.model.normalizeX else None,
            dataset.trainY.mean(dim=0) if config.model.normalizeY else None,
            dataset.trainY.std(dim=0)  if config.model.normalizeY else None
        )
    )
