from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import TrainConfig, ModelConfig
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

class MLP(nn.Module):
    def __init__(self, device, input_dim, output_dim, config: ModelConfig, normalize: Normalization, checkpoint: Optional[torch.jit.ScriptModule]):
        super().__init__()
        self.device = device
        normalize.to(device)
        self.normalizeX = normalize.X()
        self.normalizeY = normalize.Y()

        if checkpoint:
            self.model = checkpoint.model
            self.to(device)
            return

        layers = []
        fan_in = input_dim
        for (i, layer_dim) in enumerate(config.hidden_layers):
            layers.append(nn.Linear(fan_in, layer_dim, dtype=torch.float64, bias=config.bias))
            fan_in = layer_dim

            if config.batchnorm:
                layers.append(nn.BatchNorm1d(layer_dim, dtype=torch.float64))

            layers.append(activation_function(config.activation))

            if i < len(config.dropout):
                layers.append(nn.Dropout(config.dropout[i]))
        layers.append(nn.Linear(fan_in, output_dim, dtype=torch.float64, bias=config.bias))

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

def create_model(config: TrainConfig, dataset: Dataset, checkpoint: Optional[torch.jit.ScriptModule] = None):
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
        ),
        checkpoint
    )
