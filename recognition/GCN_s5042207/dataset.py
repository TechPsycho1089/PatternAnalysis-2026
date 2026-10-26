"""
Data loading and preprocessing for the Facebook Large Page-Page Network.

Each node is a verified Facebook page, each edge is a mutual like between two
pages, and each page belongs to one of four classes (see CLASS_NAMES).

Data source: the modified PyG FacebookPagePage file prepared by Gayanku
    https://github.com/gayanku/facebook_page-page_dataset
Original dataset: B. Rozemberczki, C. Allen and R. Sarkar, "Multi-scale
Attributed Node Embedding", Journal of Complex Networks, 2021.
"""
import os

import numpy as np
import torch
from torch_geometric.data import Data, download_url
from torch_geometric.utils import coalesce, remove_self_loops

DATA_URL = ("https://raw.githubusercontent.com/gayanku/"
            "facebook_page-page_dataset/main/facebook.npz")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NPZ_PATH = os.path.join(DATA_DIR, "facebook.npz")

# Label index -> class name, as documented in the Gayanku repository
CLASS_NAMES = ["tvshow", "government", "company", "politician"]


def load_graph(path=NPZ_PATH):
    """Load the Facebook page-page graph as a PyG Data object.

    Downloads the file on first use. Self-loops and duplicate edges are
    removed (GCNConv adds its own self-loops, and duplicates would make some
    neighbours count twice when averaging).

    Returns a Data object with:
        x          [N, 128] float  node features
        edge_index [2, E]   long   edges as (source, target) node index pairs
        y          [N]      long   class label of each node (0-3)
    """
    if not os.path.exists(path):
        download_url(DATA_URL, os.path.dirname(path))

    # allow_pickle=False: we never read the 'page_name' array, which is the
    # only one that needs pickle, and unpickling downloaded data is unsafe
    raw = np.load(path, allow_pickle=False)

    x = torch.from_numpy(raw["X"]).float()
    y = torch.from_numpy(raw["y"]).long()

    # The file stores edges as two separate lists (sources, targets);
    # PyG wants them stacked into a single [2, E] tensor
    edge_index = torch.stack([
        torch.from_numpy(raw["edges_x"]),
        torch.from_numpy(raw["edges_y"]),
    ]).long()

    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=x.size(0))  # sort + dedupe

    return Data(x=x, edge_index=edge_index, y=y)


if __name__ == "__main__":
    data = load_graph()
    print(data)
    print("undirected:", data.is_undirected())
    print("class counts:", torch.bincount(data.y).tolist())
