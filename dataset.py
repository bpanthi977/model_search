"""Dataset class and load_dataset."""
import torch
import h5py
import numpy as np
from config import DatasetConfig

class Dataset():
    """
    Storage class for train and validation data.

    The data are torch.tensor
    """

    def __init__(self, trainX, trainY, validateX, validateY):
        """Initialize."""
        self.trainX = trainX
        self.trainY = trainY
        self.validateX = validateX
        self.validateY = validateY

    def input_dim(self):
        """Input dimension."""
        return self.trainX.shape[-1]

    def output_dim(self):
        """Output dimension."""
        return self.trainY.shape[-1]

def get_label(db, labels):
    """Return the first group/label available in the h5 db file."""
    for label in labels:
        if label in db:
            return label

    return False

def load_dataset(config: DatasetConfig):
    """
    Load the dataset stored as torch.tensors in `cpu`.

    If train and validate labels are not separately defined
    same data is used for both.
    """
    db = h5py.File(config.db_file)
    label = config.label

    train = get_label(db, [label+'_train', label])
    if not train:
        raise ValueError(f"label {label} or {label}_train not found in db file")

    train_X = torch.from_numpy(np.array(db[train]['input'], dtype=np.float32))
    train_Y = torch.from_numpy(np.array(db[train]['output'], dtype=np.float32))

    val = get_label(db, [label+'_validate', label])
    if val == train:
        validate_X = train_X
        validate_Y = train_Y
    elif val:
        validate_X = torch.from_numpy(np.array(db[val]['input'], dtype=np.float32))
        validate_Y = torch.from_numpy(np.array(db[val]['output'], dtype=np.float32))
    else:
        raise ValueError(f"label {label} or {label}_validate not found in db file")

    # X shape is [iterations, nodes/elements, input_vec]
    # Y shape is [iterations, nodes/elements, output_vec]

    # Merge iterations and node/elements dimension
    input_dim = train_X.shape[-1]
    train_X = train_X.reshape([-1, input_dim])
    validate_X = validate_X.reshape([-1, input_dim])

    output_dim = train_Y.shape[-1]
    train_Y = train_Y.reshape([-1, output_dim])
    validate_Y = validate_Y.reshape([-1, output_dim])

    return Dataset(train_X, train_Y, validate_X, validate_Y)
