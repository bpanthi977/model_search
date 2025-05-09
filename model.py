from typing import List
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from config import TrainConfig
from dataset import Dataset

class MLP(nn.Module):
    def __init__(self, device: torch.device, dimensions: List[int], activation_function):
        super().__init__()
        self.device = device
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            if i < len(dimensions) - 2:
                layers.append(activation_function())

        self.model = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        return self.model(x)

    def get_device(self):
        return self.device

def create_model(config: TrainConfig, dataset: Dataset):
    layers = config.hidden_layers
    input_dim = dataset.input_dim()
    output_dim = dataset.output_dim()
    dimensions = [input_dim] + layers + [output_dim]
    device = torch.device(config.device)

    activations = {
        'relu': nn.ReLU
    }

    if config.activation not in activations:
        raise ValueError(f"Activation {config.activation} not supported. Use one of: {list(activations.keys())}")

    activation = activations[config.activation]
    model = MLP(device, dimensions, activation)
    return model


def train(model: MLP, dataset: Dataset, config: TrainConfig):
    dataset = TensorDataset(torch.from_numpy(dataset.X), torch.from_numpy(dataset.Y))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    losses = []
    progress_bar = tqdm(range(config.epoch), unit="epoch")
    device = model.get_device()

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    else:
        raise ValueError(f"Optimizer {config.optimizer} not supported")

    for _ in progress_bar:
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()
            Y_pred = model.forward(batch_X)
            loss = torch.sum((batch_Y - Y_pred) ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        progress_bar.set_postfix({"loss": f"{torch.sqrt(torch.tensor(avg_epoch_loss)).item():.4f}"})

def evaluate_model(model: MLP, dataset: Dataset, batch_size: int):
    "Returns Mean Squared Error of model on (X,Y) dataset"
    dataset = TensorDataset(torch.from_numpy(dataset.X), torch.from_numpy(dataset.Y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = model.get_device()

    loss = 0
    for batch_X, batch_Y in dataloader:
        X = batch_X.to(device)
        Y = batch_Y.to(device)
        Y_pred = model.forward(X)
        loss += torch.sqrt(torch.sum((Y - Y_pred)**2)).item()

    return loss
