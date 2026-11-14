import torch
from torch import nn


def degree_normalize(adj, eps=1e-8):
    return adj * (1.0 / (adj.sum(-1, keepdim=True) + eps))


class SAGEBlock(nn.Module):
    def __init__(self, dim, dropout=0):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim, bias=False)
        self.lin_neigh = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mean):
        neighbors = torch.bmm(adj_mean, self.lin_neigh(x))
        update = self.mlp(self.lin_self(x) + neighbors)
        return x + self.dropout(self.ln(update))


class GraphModel(nn.Module):
    def __init__(self, max_node_num, atom_attr_dim, latent_dim1, latent_dim2):
        super().__init__()
        self.max_node_num = max_node_num
        self.hidden = int(latent_dim1)
        self.num_layers = max(2, min(8, int(round(latent_dim2))))
        self.enc = nn.Linear(atom_attr_dim, self.hidden)
        self.layers = nn.ModuleList([SAGEBlock(self.hidden) for _ in range(self.num_layers)])
        self.readout_ln = nn.Identity()
        self.t_mlp = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 2 * self.hidden))
        nn.init.zeros_(self.t_mlp[-1].weight)
        nn.init.zeros_(self.t_mlp[-1].bias)
        self.head = nn.Sequential(
            nn.Linear(self.hidden, 256), nn.GELU(), nn.Dropout(0),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0), nn.Linear(64, 1),
        )

    @staticmethod
    def _mask_from_adj(adj):
        return adj.sum(-1) > 0

    def forward(self, node_attr_matrix, adjacency_matrix, t_matrix):
        adjacency = adjacency_matrix.float()
        features = self.enc(node_attr_matrix.float())
        neighbors = degree_normalize(adjacency)
        for layer in self.layers:
            features = layer(features, neighbors)
        mask = self._mask_from_adj(adjacency).unsqueeze(-1)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        scale, shift = self.t_mlp(t_matrix.float()).chunk(2, dim=-1)
        return self.head((1 + scale) * pooled + shift)
