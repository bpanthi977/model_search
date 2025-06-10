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
        m.weight, m.bias
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

        else:
            assert false, "Impossible initialization parameter. Should have been checked in ModelConfig."

class MLP(nn.Module):
    def __init__(self, device, input_dim, output_dim, config: ModelConfig):
        super().__init__()
        self.device = device

        layers = []
        fan_in = input_dim
        for (i, layer_dim) in enumerate(config.hidden_layers):
            layers.append(nn.Linear(fan_in, layer_dim, dtype=torch.float64))
            fan_in = layer_dim

            layers.append(activation_function(config.activation))
            if i < len(config.dropout):
                layers.append(nn.Dropout(config.dropout[i]))
        layers.append(nn.Linear(fan_in, output_dim, dtype=torch.float64))

        self.model = nn.Sequential(*layers)
        self.model.apply(lambda m: init_weights(m, config))
        self.to(device)

    def forward(self, x):
        return self.model(x)

    def get_device(self):
        return self.device

def create_model(config: TrainConfig, dataset: Dataset):
    return MLP(
        torch.device(config.device),
        dataset.input_dim(),
        dataset.output_dim(),
        config.model
    )
