import sys
import h5py
import torch
import pyrallis
import argparse
import shutil
from pathlib import Path
from config import Config
from model import create_model

def get_dims(db_file, label):
    with h5py.File(db_file, 'r') as db:
        train_label = None
        for l in [label+'_train', label]:
            if l in db:
                train_label = l
                break
        if not train_label:
             raise ValueError(f"label {label} or {label}_train not found in {db_file}")

        # Shape is [iterations, nodes/elements, vec] or [iterations, vec]
        # load_dataset logic: input_dim = train_X.shape[-1]
        input_dim = db[train_label]['input'].shape[-1]
        output_dim = db[train_label]['output'].shape[-1]
    return input_dim, output_dim

def verify_run(run_dir):
    config_path = run_dir / "config.yaml"
    model_path = run_dir / "model.pt"

    if not config_path.exists() or not model_path.exists():
        return None, "Missing config or model file"

    try:
        # Load config without system arguments interference
        config = pyrallis.parse(config_class=Config, config_path=config_path, args=[])

        # Get dimensions
        input_dim, output_dim = get_dims(config.dataset.db_file, config.dataset.label)

        # Create a mock dataset object with expected attributes
        class MockDataset:
            def __init__(self, idim, odim):
                self._idim = idim
                self._odim = odim
                # Dummy tensors for normalization calculation
                self.trainX = torch.zeros((2, idim), dtype=torch.float64)
                self.trainY = torch.zeros((2, odim), dtype=torch.float64)
            def input_dim(self): return self._idim
            def output_dim(self): return self._odim

        mock_dataset = MockDataset(input_dim, output_dim)

        # Set device to cpu for verification
        config.train.device = 'cpu'

        # This will use the current (fixed) MLP implementation
        expected_model = create_model(config.train, mock_dataset)

        # Load the saved model (TorchScript)
        saved_model = torch.jit.load(model_path, map_location='cpu')

        # Check parameter count
        expected_params = sum(p.numel() for p in expected_model.parameters())
        saved_params = sum(p.numel() for p in saved_model.parameters())

        if expected_params != saved_params:
            return False, f"Param count mismatch: Expected {expected_params}, Saved {saved_params}"

        # Compare state dicts
        expected_model.load_state_dict(saved_model.state_dict(), strict=True)
        return True, "Architectures match"
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify study runs and optionally delete faulty ones.")
    parser.add_argument("study_dir", type=str, help="Path to the study directory")
    parser.add_argument("--delete", action="store_true", help="Delete runs that fail verification")
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    if not study_dir.exists():
        print(f"Study directory {study_dir} does not exist")
        sys.exit(1)

    if (study_dir / "config.yaml").exists():
        runs = [study_dir]
    else:
        runs = [d for d in study_dir.iterdir() if d.is_dir()]
    runs.sort()

    print(f"{'Run Name':35} {'Status':10} {'Details'}")
    print("-" * 80)
    for run in runs:
        ok, msg = verify_run(run)
        if ok is None:
            continue

        if not ok and args.delete:
            status = "[DELETED]"
            shutil.rmtree(run)
        else:
            status = "[OK]" if ok else "[NOT OK]"

        # Filter out common noise from msg if possible or just print it
        detail = msg if not ok else ""
        print(f"{run.name:35} {status:12} {detail}")
