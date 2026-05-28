from pathlib import Path

import numpy as np
import pandas as pd


def _load_csv_targets(data_dirs):
    batches = []
    for data_dir in data_dirs:
        if data_dir:
            batches.append(pd.read_csv(Path(data_dir) / "stress.csv", index_col=0))
    return pd.concat(batches)


def _load_numpy_batches(data_dirs, filename):
    batches = []
    for data_dir in data_dirs:
        if data_dir:
            batches.append(np.load(Path(data_dir) / filename))
    return np.concatenate(batches, axis=0)


def _split_train_indices(total_size, train_ratio):
    import random

    train_size = int(total_size * train_ratio)
    return random.sample(range(total_size), train_size)


def train_dataset_gen(save_name, data_dirs, train_ratio, val_ratio, ch_start, ch_end, std=True):
    from joblib import dump
    from sklearn.preprocessing import StandardScaler

    all_input = _load_numpy_batches(data_dirs, "input.npy")[:, :, :, :, ch_start:ch_end]
    all_target = _load_csv_targets(data_dirs).iloc[:, 1:7] / 1000000

    trainval_idx = _split_train_indices(len(all_target), train_ratio)
    trainval_input = all_input[trainval_idx]
    trainval_target = all_target.iloc[trainval_idx]

    import random

    val_size = int(len(trainval_input) * val_ratio)
    val_idx = random.sample(range(len(trainval_input)), val_size)

    val_input = trainval_input[val_idx]
    val_target = trainval_target.iloc[val_idx]
    train_input = np.delete(trainval_input, val_idx, axis=0)
    train_target = trainval_target.drop(index=val_target.index)

    if std:
        scaler = StandardScaler()
        scaler.fit(train_target)
        train_target = scaler.transform(train_target)
        val_target = scaler.transform(val_target)
        dump(scaler, f"std_scaler{save_name}.bin", compress=True)

    np.save(f"train_index_{save_name}", trainval_idx)
    return train_input, train_target, val_input, val_target


def test_dataset_gen(save_name, data_dirs, result_dir, ch_start, ch_end):
    if len(data_dirs) == 1:
        all_input = np.load(Path(data_dirs[0]) / "input.npy")
        all_target = pd.read_csv(Path(data_dirs[0]) / "stress.csv", index_col=0)
    else:
        all_input = _load_numpy_batches(data_dirs, "input.npy")
        all_target = _load_csv_targets(data_dirs)

    inputs = all_input[:, :, :, :, ch_start:ch_end]
    targets = all_target.iloc[:, 1:7] / 1000000
    trainval_idx = np.load(Path(result_dir) / f"train_index_{save_name}.npy")
    trainval_target = targets.iloc[trainval_idx]
    return np.delete(inputs, trainval_idx, axis=0), targets.drop(index=trainval_target.index)


def test_dataset_gen_15_45(save_name, data_dirs, result_dir, ch_start, ch_end):
    if len(data_dirs) == 1:
        all_input = np.load(Path(data_dirs[0]) / "input.npy")
        all_target = pd.read_csv(Path(data_dirs[0]) / "stress.csv", index_col=0)
    else:
        all_input = _load_numpy_batches(data_dirs, "input.npy")
        all_target = _load_csv_targets(data_dirs)

    inputs = all_input[:, :, :, :, ch_start:ch_end]
    targets = all_target.iloc[:, 0:6]
    trainval_idx = np.load(Path(result_dir) / f"train_index_{save_name}.npy")
    trainval_target = targets.iloc[trainval_idx]
    return np.delete(inputs, trainval_idx, axis=0), targets.drop(index=trainval_target.index)


def new_test_dataset_gen(data_dirs, ch_start, ch_end):
    all_input = _load_numpy_batches(data_dirs, "input.npy")
    all_target = _load_csv_targets(data_dirs)
    return (
        all_target.iloc[:, 0],
        all_input[:, :, :, :, ch_start:ch_end],
        all_target.iloc[:, 1:7] / 1000000,
    )


def NormalizeData(data):
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    return (data - data_min) / (data_max - data_min), data_min, data_max


def DeNormalizeData(data, data_min, data_max):
    return data * (data_max - data_min) + data_min


def train_dataset_gen_cGAN(save_name, data_dirs, train_ratio, ch_start, ch_end, norm=True):
    import random

    all_input = _load_numpy_batches(data_dirs, "input.npy")[:, :, :, :, ch_start:ch_end].astype(np.float32)
    all_target = _load_numpy_batches(data_dirs, "target.npy")

    if norm:
        all_target, data_min, data_max = NormalizeData(all_target)
    else:
        data_min = 0
        data_max = 0
    all_target = all_target.astype(np.float32)

    trainval_idx = _split_train_indices(len(all_target), train_ratio)
    train_input = all_input[trainval_idx]
    train_target = all_target[trainval_idx]
    test_input = np.delete(all_input, trainval_idx, axis=0)
    test_target = np.delete(all_target, trainval_idx, axis=0)

    val_idx = random.sample(range(len(train_target)), 0)
    train_input = np.delete(train_input, val_idx, axis=0)
    train_target = np.delete(train_target, val_idx, axis=0)

    np.save(f"train_index_{save_name}", trainval_idx)
    return train_input, train_target, test_input, test_target, data_min, data_max


def test_dataset_gen_cGAN(save_name, data_dirs, result_dir):
    all_input = _load_numpy_batches(data_dirs, "input.npy")[:, :, :, :, 1:5].astype(np.float32)
    all_target, data_min, data_max = NormalizeData(_load_numpy_batches(data_dirs, "target.npy"))
    trainval_idx = np.load(Path(result_dir) / f"train_index_{save_name}.npy")
    return np.delete(all_input, trainval_idx, axis=0), np.delete(all_target.astype(np.float32), trainval_idx, axis=0)


def remove_suffix(input_string, suffix):
    text = str(input_string)
    return text[: -len(suffix)] if suffix and text.endswith(suffix) else text


def remove_prefix(input_string, prefix):
    text = str(input_string)
    return text[len(prefix):] if prefix and text.startswith(prefix) else text


def move_data():
    import glob
    import shutil

    data_types = ["strain", "strain-eq", "strain-pl", "strain-pl-eq"]
    typeid = 1
    source_root = Path("/home/xiao/projects/inverse_mat_des/Simulation/strain_recal")
    target_root = Path("/home/xiao/projects/inverse_mat_des/ML/dataset")
    for grain in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        target_dir = target_root / f"{grain}_cGAN" / data_types[typeid]
        for source_pattern, prefix_len in [
            (source_root / f"{grain}" / "**" / f"*{data_types[typeid]}.step2", 3),
            (source_root / f"{grain}_2" / "**" / f"*{data_types[typeid]}.step2", 5),
        ]:
            for source in glob.glob(str(source_pattern), recursive=True):
                source_text = str(source)
                suffix = f"_sim.sim/results/elts/{data_types[typeid]}/{data_types[typeid]}.step2"
                name = remove_suffix(source_text[(len(str(source_root)) + prefix_len):], suffix).replace("/", "_")
                shutil.copy(source, target_dir / f"{name}_{data_types[typeid]}.step2")
