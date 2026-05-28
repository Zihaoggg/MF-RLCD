from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class GraphDataSet(Dataset):
    def __init__(self, num_data, graph_seq):
        max_node = 300
        num_features = 5
        adjacency, features, fields, labels = [], [], [], []

        for idx in graph_seq[:num_data]:
            neighbor, feature, property_data = _load_graph_files(idx)
            feature = manipulate_feature(feature, max_node, num_features)
            neighbor = normalize_adj(neighbor, max_node)
            property_data = property_data[property_data.min(axis=1) >= 0, :]
            t_values = np.delete(property_data, 1, axis=1)
            y_values = np.delete(property_data, 0, axis=1)

            adjacency.extend([neighbor] * len(t_values))
            features.extend([feature] * len(t_values))
            fields.extend(t_values)
            labels.extend(y_values)

        fields, labels = normalize_t_label(np.asarray(fields), np.asarray(labels))
        self.adjacency_matrix = np.asarray(adjacency, dtype=object)
        self.node_attr_matrix = np.asarray(features, dtype=object)
        self.t_matrix = np.asarray(fields)
        self.label_matrix = np.asarray(labels)

        print("--------------------")
        print("Training Data:")
        print("adjacency matrix:\t", self.adjacency_matrix.shape)
        print("node attribute matrix:\t", self.node_attr_matrix.shape)
        print("t matrix:\t\t", self.t_matrix.shape)
        print("label name:\t\t", self.label_matrix.shape)
        print("--------------------")

    def __len__(self):
        return len(self.adjacency_matrix)

    def __getitem__(self, idx):
        adjacency = torch.from_numpy(self.adjacency_matrix[idx].toarray())
        node_attr = torch.from_numpy(self.node_attr_matrix[idx].toarray())
        field = torch.from_numpy(np.asarray(self.t_matrix[idx]))
        label = torch.from_numpy(np.asarray(self.label_matrix[idx]))
        return adjacency, node_attr, field, label


def _load_graph_files(index):
    folder = Path("data") / f"structure-{index}"
    return (
        np.loadtxt(folder / "neighbor.txt"),
        np.loadtxt(folder / "feature.txt"),
        np.loadtxt(folder / "property.txt"),
    )


def normalize_adj(neighbor, max_node):
    matrix = np.asarray(neighbor, dtype=np.float64).copy()
    np.fill_diagonal(matrix, 1.0)
    degree = matrix.sum(axis=0)
    degree[degree == 0] = 1.0
    inv_sqrt = np.diag(np.power(degree, -0.5))
    matrix = inv_sqrt @ matrix @ inv_sqrt
    padded = np.zeros((max_node, max_node), dtype=np.float64)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix
    return sparse.csr_matrix(padded)


def manipulate_feature(feature, max_node, features):
    values = np.delete(np.asarray(feature, dtype=np.float64), 0, axis=1)
    std = np.std(values[:, 3])
    if std == 0:
        values[:, 3] = 0
    else:
        values[:, 3] = (values[:, 3] - np.mean(values[:, 3])) / std
    padded = np.zeros((max_node, features), dtype=np.float64)
    padded[: values.shape[0], : values.shape[1]] = values
    return sparse.csr_matrix(padded)


def normalize_t_label(t_matrix, label_matrix):
    label_mean = np.mean(label_matrix)
    label_std = np.std(label_matrix)
    if label_std == 0:
        label_std = 1.0
    normalized = (label_matrix - label_mean) / label_std
    np.savez_compressed("norm.npz", norm=np.array([label_mean, label_std]))
    return t_matrix, normalized


def get_data(batch_size, idx_path, validation_index, testing_index, folds, num_data):
    split_data = np.load(idx_path, allow_pickle=True)
    indices = split_data["indices"]
    graph_seq = split_data["graph_seq"]
    validation_idx = indices[validation_index]
    test_idx = indices[testing_index]
    train_idx = indices[[i for i in range(folds) if i not in {validation_index, testing_index}]]
    train_idx = [item for fold in train_idx for item in fold]

    dataset = GraphDataSet(num_data, graph_seq)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_idx))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))
    return train_loader, validation_loader, test_loader
