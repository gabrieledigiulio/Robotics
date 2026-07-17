import gzip
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def load_raw_data(dataset_dir: Path, trial: int, condition: str):
    """
    Loads raw data from a gzipped pickle file for a specific trial and condition.
    """
    path = dataset_dir / f"trial_{trial}_{condition}.pkl"

    with gzip.open(str(path), "rb") as f:
        _time   = pickle.load(f)
        _pos    = pickle.load(f)
        _force  = pickle.load(f)
        actions = pickle.load(f)
        vision  = pickle.load(f)

    return vision, actions, _force, _pos


def make_temporal_splits(vision: np.ndarray,
                         actions: np.ndarray,
                         ratios: tuple = (0.70, 0.15, 0.15)):
    """
    Splits the temporal data into training, validation, and testing sets.
    """
    assert len(ratios) == 3, "ratios must have 3 elements: (train, val, test)"
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    X_t  = vision[:-1].astype(np.float32) / 255.0
    X_t1 = vision[1:].astype(np.float32)  / 255.0
    A_t  = actions[:-1].astype(np.float32)

    X_t  = np.transpose(X_t,  (0, 3, 1, 2))
    X_t1 = np.transpose(X_t1, (0, 3, 1, 2))

    N         = len(X_t)
    train_end = int(N * ratios[0])
    val_end   = int(N * (ratios[0] + ratios[1]))

    slices = {
        "train": slice(0,         train_end),
        "val":   slice(train_end, val_end),
        "test":  slice(val_end,   N),
    }

    scaler  = StandardScaler()
    A_train = scaler.fit_transform(A_t[slices["train"]])
    A_val   = scaler.transform(A_t[slices["val"]])
    A_test  = scaler.transform(A_t[slices["test"]])

    scaled_A = {"train": A_train, "val": A_val, "test": A_test}

    splits = {
        name: (X_t[slices[name]], X_t1[slices[name]], scaled_A[name])
        for name in ("train", "val", "test")
    }

    total = sum(len(v[0]) for v in splits.values())
    for name, (x, _, a) in splits.items():
        pct = 100.0 * len(x) / total
        print(f"  [Split] {name:5s}: {len(x):5d} samples  ({pct:.0f}%)")

    return splits, scaler


def load_and_concat_datasets(dataset_dir: Path,
                              dataset_list: list,
                              ratios: tuple = (0.70, 0.15, 0.15)):
    """
    Loads multiple datasets, splits them temporally according to the given ratios,
    and concatenates the splits across all datasets.
    """
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6, "ratios must have 3 elements and sum to 1.0"

    per_split_Xt  = {"train": [], "val": [], "test": []}
    per_split_Xt1 = {"train": [], "val": [], "test": []}
    per_split_St  = {"train": [], "val": [], "test": []}
    per_split_St1 = {"train": [], "val": [], "test": []}
    per_split_Pt  = {"train": [], "val": [], "test": []}
    per_split_Pt1 = {"train": [], "val": [], "test": []}
    per_split_At  = {"train": [], "val": [], "test": []}

    for trial, condition in dataset_list:
        vision, actions, force, pos = load_raw_data(dataset_dir, trial, condition)

        Xt  = vision[:-1].astype(np.float32) / 255.0
        Xt1 = vision[1:].astype(np.float32)  / 255.0
        St  = force[:-1].astype(np.float32)
        St1 = force[1:].astype(np.float32)
        Pt  = pos[:-1].astype(np.float32)
        Pt1 = pos[1:].astype(np.float32)
        At  = actions[:-1].astype(np.float32)

        Xt  = np.transpose(Xt,  (0, 3, 1, 2))
        Xt1 = np.transpose(Xt1, (0, 3, 1, 2))

        n         = len(Xt)
        train_end = int(n * ratios[0])
        val_end   = int(n * (ratios[0] + ratios[1]))

        trial_slices = {
            "train": slice(0,         train_end),
            "val":   slice(train_end, val_end),
            "test":  slice(val_end,   n),
        }

        for name, sl in trial_slices.items():
            per_split_Xt[name].append(Xt[sl])
            per_split_Xt1[name].append(Xt1[sl])
            per_split_St[name].append(St[sl])
            per_split_St1[name].append(St1[sl])
            per_split_Pt[name].append(Pt[sl])
            per_split_Pt1[name].append(Pt1[sl])
            per_split_At[name].append(At[sl])

        print(f"  [Split] trial_{trial}_{condition}: "
              f"train={train_end}  val={val_end - train_end}  "
              f"test={n - val_end}")

    X_t  = {name: np.concatenate(chunks, axis=0) for name, chunks in per_split_Xt.items()}
    X_t1 = {name: np.concatenate(chunks, axis=0) for name, chunks in per_split_Xt1.items()}
    S_t_raw  = {name: np.concatenate(chunks, axis=0) for name, chunks in per_split_St.items()}
    S_t1_raw = {name: np.concatenate(chunks, axis=0) for name, chunks in per_split_St1.items()}
    P_t_raw  = {name: np.concatenate(chunks, axis=0) for name, chunks in per_split_Pt.items()}
    P_t1_raw = {name: np.concatenate(chunks, axis=0) for name, chunks in per_split_Pt1.items()}
    A_raw = {name: np.concatenate(chunks, axis=0) for name, chunks in per_split_At.items()}

    total_samples = sum(len(X_t[name]) for name in X_t)

    scaler = StandardScaler()
    scaled_A = {
        "train": scaler.fit_transform(A_raw["train"]),
        "val":   scaler.transform(A_raw["val"]),
        "test":  scaler.transform(A_raw["test"]),
    }

    force_scaler = StandardScaler()
    scaled_S = {
        "train": force_scaler.fit_transform(S_t_raw["train"]),
        "val":   force_scaler.transform(S_t_raw["val"]),
        "test":  force_scaler.transform(S_t_raw["test"]),
    }
    scaled_S1 = {
        "train": force_scaler.transform(S_t1_raw["train"]),
        "val":   force_scaler.transform(S_t1_raw["val"]),
        "test":  force_scaler.transform(S_t1_raw["test"]),
    }

    proprio_scaler = StandardScaler()
    scaled_P = {
        "train": proprio_scaler.fit_transform(P_t_raw["train"]),
        "val":   proprio_scaler.transform(P_t_raw["val"]),
        "test":  proprio_scaler.transform(P_t_raw["test"]),
    }
    scaled_P1 = {
        "train": proprio_scaler.transform(P_t1_raw["train"]),
        "val":   proprio_scaler.transform(P_t1_raw["val"]),
        "test":  proprio_scaler.transform(P_t1_raw["test"]),
    }

    splits = {
        name: (X_t[name], X_t1[name], scaled_S[name], scaled_S1[name], scaled_P[name], scaled_P1[name], scaled_A[name])
        for name in ("train", "val", "test")
    }

    per_dataset_splits = []
    for i, (trial, condition) in enumerate(dataset_list):
        per_dataset_splits.append({
            "trial": trial,
            "condition": condition,
            "test": (
                per_split_Xt["test"][i],
                per_split_Xt1["test"][i],
                force_scaler.transform(per_split_St["test"][i]),
                force_scaler.transform(per_split_St1["test"][i]),
                proprio_scaler.transform(per_split_Pt["test"][i]),
                proprio_scaler.transform(per_split_Pt1["test"][i]),
                scaler.transform(per_split_At["test"][i])
            )
        })

    for name, (x, _, _, _, _, _, _) in splits.items():
        pct = 100.0 * len(x) / total_samples
        print(f"  [Split] {name:5s}: {len(x):5d} samples  ({pct:.0f}%)")

    return splits, scaler, force_scaler, proprio_scaler, per_dataset_splits


class WorldModelDataset(Dataset):
    """
    PyTorch Dataset wrapper for the World Model data.
    """

    def __init__(self, X_t: np.ndarray, X_t1: np.ndarray, S_t: np.ndarray, S_t1: np.ndarray, P_t: np.ndarray, P_t1: np.ndarray, A_t: np.ndarray):
        self.X_t  = torch.tensor(X_t,  dtype=torch.float32)
        self.X_t1 = torch.tensor(X_t1, dtype=torch.float32)
        self.S_t  = torch.tensor(S_t,  dtype=torch.float32)
        self.S_t1 = torch.tensor(S_t1, dtype=torch.float32)
        self.P_t  = torch.tensor(P_t,  dtype=torch.float32)
        self.P_t1 = torch.tensor(P_t1, dtype=torch.float32)
        self.A_t  = torch.tensor(A_t,  dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X_t)

    def __getitem__(self, idx: int):
        return self.X_t[idx], self.X_t1[idx], self.S_t[idx], self.S_t1[idx], self.P_t[idx], self.P_t1[idx], self.A_t[idx]