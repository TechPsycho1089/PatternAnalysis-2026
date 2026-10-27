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
from torch_geometric.utils import coalesce, degree, homophily, remove_self_loops

DATA_URL = ("https://raw.githubusercontent.com/gayanku/"
            "facebook_page-page_dataset/main/facebook.npz")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NPZ_PATH = os.path.join(DATA_DIR, "facebook.npz")

# Label index -> class name, as documented in the Gayanku repository
CLASS_NAMES = ["tvshow", "government", "company", "politician"]


def load_graph(path=NPZ_PATH, verbose=False):
    """Load the Facebook page-page graph as a PyG Data object.

    Downloads the file on first use. Self-loops and duplicate edges are
    removed (GCNConv adds its own self-loops, and duplicates would make some
    neighbours count twice when averaging). With verbose=True, prints how
    many edges each cleaning step removed.

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

    n_raw = edge_index.size(1)
    is_loop = edge_index[0] == edge_index[1]
    n_self_loops = int(is_loop.sum())
    n_loop_pages = edge_index[0, is_loop].unique().numel()
    edge_index, _ = remove_self_loops(edge_index)
    n_after_loops = edge_index.size(1)
    edge_index = coalesce(edge_index, num_nodes=x.size(0))  # sort + dedupe

    if verbose:
        print(f"edges in file:              {n_raw}")
        print(f"self-loop entries removed:  {n_self_loops} "
              f"(on {n_loop_pages} distinct pages)")
        print(f"duplicate edges removed:    {n_after_loops - edge_index.size(1)}")
        print(f"directed edges kept:        {edge_index.size(1)}")

    return Data(x=x, edge_index=edge_index, y=y)


def describe_graph(data):
    """Print a data audit: size, class balance, degree and homophily.

    Homophily uses every node's label, including test nodes. This is fine
    for describing the dataset, but these numbers must never be used to
    train or tune a model.
    """
    n = data.num_nodes
    print(f"nodes: {n}   undirected edges: {data.edge_index.size(1) // 2}   "
          f"features per node: {data.num_features}")
    counts = torch.bincount(data.y, minlength=len(CLASS_NAMES))
    for name, c in zip(CLASS_NAMES, counts.tolist()):
        print(f"  {name:11s} {c:6d}  ({c / n:.1%})")

    deg = degree(data.edge_index[0], n)
    print(f"degree: min {int(deg.min())}  median {int(deg.median())}  "
          f"mean {deg.mean():.1f}  max {int(deg.max())}  "
          f"isolated nodes {int((deg == 0).sum())}")

    for method in ("edge", "node", "edge_insensitive"):
        h = homophily(data.edge_index, data.y, method=method)
        print(f"homophily ({method}): {h:.3f}")


def stratified_split(y, train_frac=0.6, val_frac=0.2, seed=42):
    """Split nodes into train/val/test masks, keeping the class mix equal.

    Each class is shuffled and cut separately, so every split has the same
    proportion of each class (important for the smaller politician class).
    The remaining 1 - train_frac - val_frac of each class goes to test.
    The same seed always gives the same split.

    Returns three boolean masks of shape [N]: True = node is in that split.
    """
    generator = torch.Generator().manual_seed(seed)
    n = y.size(0)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    for c in y.unique():
        idx = (y == c).nonzero(as_tuple=True)[0]                 # nodes of class c
        idx = idx[torch.randperm(idx.numel(), generator=generator)]  # shuffle
        n_train = round(train_frac * idx.numel())
        n_val = round(val_frac * idx.numel())
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

    # Leakage guard: every node must be in exactly one split
    in_splits = train_mask.int() + val_mask.int() + test_mask.int()
    assert (in_splits == 1).all(), "train/val/test masks overlap or miss nodes"

    return train_mask, val_mask, test_mask


def load_data(seed=42):
    """Load the graph and attach train/val/test masks (used by train.py)."""
    data = load_graph()
    data.train_mask, data.val_mask, data.test_mask = stratified_split(
        data.y, seed=seed)
    return data


if __name__ == "__main__":
    print("=== Cleaning ===")
    data = load_graph(verbose=True)
    print("\n=== Data audit ===")
    describe_graph(data)
    print("undirected:", data.is_undirected())

    print("\n=== Split ===")
    data.train_mask, data.val_mask, data.test_mask = stratified_split(data.y)
    for name in ["train_mask", "val_mask", "test_mask"]:
        mask = data[name]
        counts = torch.bincount(data.y[mask], minlength=len(CLASS_NAMES))
        print(f"{name:10s} {int(mask.sum()):6d} nodes  per class: {counts.tolist()}")
