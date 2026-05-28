import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from graph_data import get_data
from tensor_utils import macro_avg_err, tensor_to_variable, variable_to_numpy


def degree_normalize(adj, eps=1e-8):
    degree = adj.sum(-1, keepdim=True)
    return adj / (degree + eps)


class SAGEBlock(nn.Module):
    def __init__(self, dim, dropout=0):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim, bias=False)
        self.lin_neigh = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mean):
        h_self = self.lin_self(x)
        h_nei = torch.bmm(adj_mean, self.lin_neigh(x))
        h = self.mlp(h_self + h_nei)
        h = self.ln(h)
        h = self.dropout(h)
        return x + h


class GraphModel(nn.Module):
    def __init__(self, max_node_num, atom_attr_dim, latent_dim1, latent_dim2):
        super().__init__()
        self.max_node_num = max_node_num
        self.hidden = int(latent_dim1)
        self.num_layers = int(max(2, min(8, int(round(latent_dim2)))))
        self.enc = nn.Linear(atom_attr_dim, self.hidden)
        self.layers = nn.ModuleList([SAGEBlock(self.hidden, dropout=0) for _ in range(self.num_layers)])
        self.readout_ln = nn.Identity()
        self.t_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 2 * self.hidden),
        )
        nn.init.zeros_(self.t_mlp[-1].weight)
        nn.init.zeros_(self.t_mlp[-1].bias)
        self.head = nn.Sequential(
            nn.Linear(self.hidden, 256),
            nn.GELU(),
            nn.Dropout(0),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _mask_from_adj(adj):
        return adj.sum(-1) > 0

    def forward(self, node_attr_matrix, adjacency_matrix, t_matrix):
        x = self.enc(node_attr_matrix.float())
        adj_mean = degree_normalize(adjacency_matrix.float())
        for layer in self.layers:
            x = layer(x, adj_mean)
        mask = self._mask_from_adj(adjacency_matrix.float()).unsqueeze(-1)
        graph_feature = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        gamma, beta = torch.chunk(self.t_mlp(t_matrix.float()), 2, dim=-1)
        graph_feature = (1.0 + gamma) * graph_feature + beta
        return self.head(graph_feature)


def train(model, train_data_loader, validation_data_loader, epochs, checkpoint_dir, optimizer, criterion, validation_index, folder_name):
    print()
    print("*** Training started! ***")
    print()

    output_path = os.path.join(folder_name, f"learning_Output_{validation_index}.txt")
    with open(output_path, "w", encoding="utf-8") as output:
        print("Epoch Training_time Training_MSE Validation_MSE", file=output, flush=True)
        for epoch in range(epochs):
            model.train()
            start = time.time()
            for adjacency_matrix, node_attr_matrix, t_matrix, label_matrix in train_data_loader:
                adjacency_matrix = tensor_to_variable(adjacency_matrix)
                node_attr_matrix = tensor_to_variable(node_attr_matrix)
                t_matrix = tensor_to_variable(t_matrix)
                label_matrix = tensor_to_variable(label_matrix)
                optimizer.zero_grad()
                y_pred = model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix, t_matrix=t_matrix)
                loss = criterion(y_pred, label_matrix)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_mse = test(model, train_data_loader, "Training", False, criterion, validation_index, folder_name)[1]
            validation_mse = test(model, validation_data_loader, "Validation", False, criterion, validation_index, folder_name)[1]
            print("%d %.3f %e %e" % (epoch, time.time() - start, train_mse, validation_mse), file=output, flush=True)


def test(model, data_loader, test_val_tr, printcond, criterion, running_index, folder_name):
    model.eval()
    if data_loader is None:
        return None, None

    y_true, y_pred_all = [], []
    with torch.no_grad():
        for adjacency_matrix, node_attr_matrix, t_matrix, label_matrix in data_loader:
            adjacency_matrix = tensor_to_variable(adjacency_matrix)
            node_attr_matrix = tensor_to_variable(node_attr_matrix)
            t_matrix = tensor_to_variable(t_matrix)
            label_matrix = tensor_to_variable(label_matrix)
            y_pred = model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix, t_matrix=t_matrix)
            y_true.extend(variable_to_numpy(label_matrix))
            y_pred_all.extend(variable_to_numpy(y_pred))

    label_mean, label_std = np.load("norm.npz", allow_pickle=True)["norm"]
    y_true = np.asarray(y_true) * label_std + label_mean
    y_pred_all = np.asarray(y_pred_all) * label_std + label_mean
    total_loss = macro_avg_err(y_pred_all, y_true)
    total_mse = criterion(torch.from_numpy(y_pred_all), torch.from_numpy(y_true)).item()

    if printcond:
        output_path = os.path.join(folder_name, f"{test_val_tr}_Output_{running_index}.txt")
        with open(output_path, "w", encoding="utf-8") as output:
            print(f"{test_val_tr} Set Predictions: ", file=output, flush=True)
            print("True_value Predicted_value", file=output, flush=True)
            for true_value, pred_value in zip(y_true, y_pred_all):
                print("%f, %f" % (true_value, pred_value), file=output, flush=True)

    return total_loss, total_mse


def _make_optimizer(name, model, learning_rate):
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    if name == "AdamW":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {name}")


def _set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_node_num", type=int, default=300)
    parser.add_argument("--atom_attr_dim", type=int, default=5)
    parser.add_argument("--num_graphs", type=int, default=492)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_learning_rate", type=float, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/")
    parser.add_argument("--validation_index", type=int, default=0)
    parser.add_argument("--testing_index", type=int, default=1)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--idx_path", type=str, default="indices_and_graphseq.npz")
    parser.add_argument("--folder_name", type=str, default="output/")
    parser.add_argument("--num_data", type=int, default=492)
    parser.add_argument("--hyper", type=int, default=101)
    return parser


def main():
    args = build_parser().parse_args()
    os.makedirs(args.checkpoint, exist_ok=True)
    os.makedirs(args.folder_name, exist_ok=True)
    _set_seed(args.seed)

    with open(f"hyper/{args.hyper}.json", "r", encoding="utf-8") as fh:
        hyper = json.load(fh)

    model = GraphModel(args.max_node_num, args.atom_attr_dim, hyper["latent_dim1"], hyper["latent_dim2"])
    if torch.cuda.is_available():
        model.cuda()

    optimizer = _make_optimizer(hyper["optim"], model, hyper["lr"])
    criterion = nn.MSELoss()
    train_loader, validation_loader, test_loader = get_data(
        args.batch_size,
        args.idx_path,
        args.validation_index,
        args.testing_index,
        args.folds,
        args.num_data,
    )

    train_start = time.time()
    train(model, train_loader, validation_loader, hyper["epoch"], args.checkpoint, optimizer, criterion, args.validation_index, args.folder_name)
    train_end = time.time()

    torch.save(model, f"{args.checkpoint}/checkpoint.pth")
    torch.save(model.state_dict(), f"{args.checkpoint}/checkpoint.state_dict.pth", _use_new_zipfile_serialization=False)
    with open(f"{args.checkpoint}/checkpoint.meta.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "max_node_num": args.max_node_num,
                "atom_attr_dim": args.atom_attr_dim,
                "latent_dim1": int(hyper["latent_dim1"]),
                "latent_dim2": int(hyper["latent_dim2"]),
            },
            fh,
        )

    train_rel, train_mse = test(model, train_loader, "Training", True, criterion, args.validation_index, args.folder_name)
    validation_rel, validation_mse = test(model, validation_loader, "Validation", True, criterion, args.validation_index, args.folder_name)
    test_start = time.time()
    test_rel, test_mse = test(model, test_loader, "Test", True, criterion, args.testing_index, args.folder_name)
    test_end = time.time()

    print("--------------------")
    print("validation_index : {}".format(args.validation_index))
    print("testing_index : {}".format(args.testing_index))
    print("training_time : {}".format(train_end - train_start))
    print("testing_time : {}".format(test_end - test_start))
    print("Train Relative Error: {:.3f}%".format(100 * train_rel))
    print("Validation Relative Error: {:.3f}%".format(100 * validation_rel))
    print("Test Relative Error: {:.3f}%".format(100 * test_rel))
    print("Train MSE : {}".format(train_mse))
    print("Validation MSE : {}".format(validation_mse))
    print("Test MSE: {}".format(test_mse))


if __name__ == "__main__":
    main()
