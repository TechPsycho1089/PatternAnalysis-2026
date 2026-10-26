"""
Model definitions: a 2-layer MLP (baseline) and a 2-layer GCN (main model).

The two models are identical except that the GCN averages each node with its
neighbours in every layer, so any difference in results comes from the graph.

Each GCN layer computes  H' = Â · H · W , where
    H · W   is an ordinary linear layer applied to every node, and
    Â       averages each node with its neighbours, using the adjacency
            matrix with self-loops added and degree normalisation:
            Â = D^(-1/2) (A + I) D^(-1/2).
Reference: T. N. Kipf and M. Welling, "Semi-supervised classification with
graph convolutional networks", ICLR 2017.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MLP(torch.nn.Module):
    """Two-layer perceptron baseline: classifies each node from its own
    features only, ignoring all edges.

    Input:  x [N, in_dim] node features, edge_index (accepted but ignored, so
            both models can be called the same way in train.py)
    Output: [N, n_classes] raw class scores (logits) for every node
    """

    def __init__(self, in_dim=128, hidden=64, n_classes=4, dropout=0.5):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden)     # 128 -> 64
        self.lin2 = torch.nn.Linear(hidden, n_classes)  # 64 -> 4
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        # Same steps as the GCN below, minus the neighbour averaging
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class GCN(torch.nn.Module):
    """Two-layer graph convolutional network for node classification.

    Input:  x [N, in_dim] node features, edge_index [2, E] edges
    Output: [N, n_classes] raw class scores (logits) for every node
    """

    def __init__(self, in_dim=128, hidden=64, n_classes=4, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)     # 128 -> 64
        self.conv2 = GCNConv(hidden, n_classes)  # 64 -> 4
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1: mix each page with its neighbours, then ReLU (the "bend")
        x = F.relu(self.conv1(x, edge_index))
        # Dropout: randomly zero half the hidden values, during training only,
        # so the model cannot rely too heavily on any single value
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2: mix again (now reaching neighbours of neighbours) and
        # produce one score per class
        return self.conv2(x, edge_index)
