"""Dataset class and load_dataset."""
import math
import torch
import h5py
import numpy as np
from config import DatasetConfig
import argparse
from pathlib import Path
from utils import get_loss_fn

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

def sample(arr: np.ndarray, fraction):
    if fraction == 1:
        return arr

    total_rows = len(arr)
    sample_count = math.ceil(total_rows * fraction)
    random_indices = np.random.choice(total_rows, sample_count, replace=False)
    return arr[random_indices]

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

    subset = config.subset if config.subset > 0 else db[train]['input'].shape[0]

    train_X = torch.from_numpy(sample(np.array(db[train]['input'][0:subset], dtype=np.float64), config.sample))
    train_Y = torch.from_numpy(sample(np.array(db[train]['output'][0:subset], dtype=np.float64), config.sample))

    val = get_label(db, [label+'_validate', label])
    if val == train:
        validate_X = train_X
        validate_Y = train_Y
    elif val:
        validate_X = torch.from_numpy(sample(np.array(db[val]['input'], dtype=np.float64), config.sample))
        validate_Y = torch.from_numpy(sample(np.array(db[val]['output'], dtype=np.float64), config.sample))
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

    if config.device == "cuda":
        device = torch.device("cuda")
        train_X = train_X.to(device)
        train_Y = train_Y.to(device)
        validate_X = validate_X.to(device)
        validate_Y = validate_Y.to(device)

    return Dataset(train_X, train_Y, validate_X, validate_Y)

def list_all_groups(h5_object):
    """
    Recursively lists all groups within an h5py object (File or Group).
    """
    groups = []

    def visitor_func(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append(name)

    h5_object.visititems(visitor_func)
    return groups

def print_stats(tensor, ref_mean=None, ref_std=None, ref_name=""):
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0) + 1e-5
    ref_mean = ref_mean if ref_mean != None else mean
    ref_std = ref_std if ref_std != None else std
    ref_name = f"(wrt {ref_name})" if ref_name else ""

    np.set_printoptions(precision=4, sign='+')
    l1 = get_loss_fn('mae')(tensor, ref_mean.unsqueeze(0).expand(tensor.shape))
    l2 = get_loss_fn('mse')(tensor, ref_mean.unsqueeze(0).expand(tensor.shape))
    sl1 = get_loss_fn('smoothl1')(tensor, ref_mean.unsqueeze(0).expand(tensor.shape))
    print(f" mean = {mean.numpy()}\n std  = {std.numpy()}")
    print(f" l1 = {l1:.4f}, smooth_l1 = {sl1:.4f} l2={l2:.4f} {ref_name}")

    normalized = (tensor - ref_mean) / ref_std
    zeros = torch.zeros_like(normalized)

    l1 = get_loss_fn('mae')(normalized, zeros)
    l2 = get_loss_fn('mse')(normalized, zeros)
    sl1 = get_loss_fn('smoothl1')(normalized, zeros)
    print(f" normalized l1 = {l1:.4f}, smooth_l1 = {sl1:.4f} l2={l2:.4f} {ref_name}")
    print(f"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dataset loader",
        description="Load dataset and tell statistics about it."
    )

    parser.add_argument("--file",help="Dataset file (.h5)")

    args = parser.parse_args()

    if not args.file:
        print(f"Dataset file not specified. --file")
        exit(1)

    dataset_file = Path(args.file)

    if not dataset_file.exists():
        print(f"File --file {dataset_file} doesn't exits.")
        exit(1)

    db = h5py.File(dataset_file)
    groups = list_all_groups(db)

    total = sum([db[group]['input'].shape[0] if 'input' in db[group] else 0 for group in groups])

    ref_mean = None
    ref_std = None
    ref_name = None
    for group in groups:
        if 'input' in db[group]:
            s = db[group]['input'].shape
            print(f"{group}/input: {s} ({s[0]/total:.2f}%)")
            tensor = torch.from_numpy(np.array(db[group]['input'], dtype=np.float64))
            print_stats(tensor)
        if 'output' in db[group]:
            s = db[group]['output'].shape
            print(f"{group}/output: {s}")
            tensor = torch.from_numpy(np.array(db[group]['output'], dtype=np.float64))
            print_stats(tensor, ref_mean, ref_std, ref_name)
            if str.endswith(group, "train"):
                ref_name = f"{group}/output"
                ref_mean = tensor.mean(dim=0)
                ref_std = tensor.std(dim=0) + 1e-5
