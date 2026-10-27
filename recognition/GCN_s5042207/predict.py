"""
Evaluate the trained MLP and GCN on the test set and analyse their confidence.

Usage:
    python predict.py      # needs the checkpoints saved by train.py for both models

Loads the saved model of every seed, predicts every page once, and reports:
  1. per-type precision, recall, F1 and AUROC, macro-F1, and confusion matrices

Numbers are mean ± std over the seeds. All settings were fixed beforehand on
the validation set, so this is the first and only use of the test labels.
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")  # save plots to files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from dataset import CLASS_NAMES, load_data
from train import HERE, MODELS


def load_models(name, data):
    """Rebuild every seed's model with the settings train.py saved, and load
    its trained weights. Returns {seed: model}."""
    with open(os.path.join(HERE, "results", f"{name}_results.json")) as f:
        args = json.load(f)["args"]
    models = {}
    for seed in args["seeds"]:
        model = MODELS[name](in_dim=data.num_features, hidden=args["hidden"],
                             n_classes=len(CLASS_NAMES), num_layers=args["layers"],
                             dropout=args["dropout"])
        path = os.path.join(HERE, "checkpoints", f"{name}_seed{seed}.pt")
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()                  # dropout OFF
        models[seed] = model
    return models


@torch.no_grad()
def predict_probs(model, data):
    """Probability of each type for every page: softmax of the 4 scores."""
    return model(data.x, data.edge_index).softmax(dim=1)


def classification_metrics(y, probs):
    """Accuracy, macro-F1, macro-AUROC and per-type precision/recall/F1/AUROC.

    AUROC (one type vs the rest) asks: if we pick a random page of this type
    and a random page of another type, how often does the model give the first
    one a higher probability for this type? 1.0 = always, 0.5 = coin flip."""
    pred = probs.argmax(dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=range(len(CLASS_NAMES)), zero_division=0)
    auroc = [roc_auc_score(y == c, probs[:, c]) for c in range(len(CLASS_NAMES))]
    metrics = {"accuracy": (pred == y).float().mean().item(),
               "macro_f1": f1_score(y, pred, average="macro"),
               "macro_auroc": statistics.mean(auroc)}
    for c, name in enumerate(CLASS_NAMES):
        metrics[f"{name}_precision"] = float(precision[c])
        metrics[f"{name}_recall"] = float(recall[c])
        metrics[f"{name}_f1"] = float(f1[c])
        metrics[f"{name}_auroc"] = auroc[c]
    return metrics


def mean_std(runs):
    """{metric: {"mean", "std"}} over a list of per-seed metric dicts."""
    return {k: {"mean": statistics.mean(r[k] for r in runs),
                "std": statistics.stdev(r[k] for r in runs) if len(runs) > 1 else 0.0}
            for k in runs[0]}


def plot_confusion(y, probs, path):
    """Row-normalised confusion matrices (rows = true type, sum to 100%),
    averaged over seeds, one panel per model."""
    fig, axes = plt.subplots(1, len(probs), figsize=(5.2 * len(probs), 4.6))
    for ax, (name, seed_probs) in zip(axes, probs.items()):
        cm = np.mean([confusion_matrix(y, p.argmax(dim=1), normalize="true")
                      for p in seed_probs.values()], axis=0) * 100
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, f"{cm[i, j]:.1f}%", ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > 50 else "#222222")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30)
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set(xlabel="predicted type", ylabel="true type", title=name.upper())
    fig.suptitle("Test confusion matrices (% of each true type, mean over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    data = load_data(seed=42)
    test = data.test_mask
    y_test = data.y[test]

    # Predict once per model and seed; later analyses reuse these probabilities
    probs = {name: {seed: predict_probs(model, data)
                    for seed, model in load_models(name, data).items()}
             for name in ("mlp", "gcn")}

    # 1. Classification quality on the test set
    report = {}
    for name, seed_probs in probs.items():
        runs = [classification_metrics(y_test, p[test]) for p in seed_probs.values()]
        report[name] = mean_std(runs)

    print("=== Test set: mean ± std over seeds ===")
    print(f"{'':22s}{'MLP':>18s}{'GCN':>18s}")
    for metric in report["mlp"]:
        cells = "".join(f"{report[m][metric]['mean']:>11.3f} ± {report[m][metric]['std']:.3f}"
                        for m in ("mlp", "gcn"))
        print(f"{metric:22s}{cells}")

    plot_confusion(y_test, {name: {s: p[test] for s, p in sp.items()}
                            for name, sp in probs.items()},
                   os.path.join(HERE, "figures", "confusion_matrices.png"))
    print("saved figures/confusion_matrices.png")

    path = os.path.join(HERE, "results", "test_metrics.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print("saved results/test_metrics.json")


if __name__ == "__main__":
    main()
