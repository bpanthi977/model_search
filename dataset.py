import h5py
import numpy as np
from config import DatasetConfig

class Dataset():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def input_dim(self):
        return self.X.shape[-1]

    def output_dim(self):
        return self.Y.shape[-1]

def load_dataset(config: DatasetConfig):
    db = h5py.File(config.db_file)
    label = config.label
    if not (label in db):
        raise ValueError(f"label {label} not found in db file")

    train_X = np.array(db[label]['input'], dtype=np.float32)
    train_Y = np.array(db[label]['output'], dtype=np.float32)

    return Dataset(train_X, train_Y)
