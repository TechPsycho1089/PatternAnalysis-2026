# Multi-layer GCN for Node Classification on the Facebook Page-Page Network

*COMP3710 Pattern Recognition Report — s5042207*

## Overview

**Problem.** Facebook has many official ("verified") pages, and a platform may need to know what kind each one is, for example to apply political-advertising rules to politicians but not to TV shows. This project sorts 22,470 Facebook pages into four types: **TV show, government organisation, company or politician**.

**Data.** Each page comes with two kinds of information:

- **Features:** 128 numbers per page, derived from the text the page owners wrote about themselves.
- **Links:** two pages are linked if they *like each other*. There are 170,823 such links.

**Approach.** A **Graph Convolutional Network (GCN)** [Kipf & Welling, 2017] uses both the features and the links. It is compared with a **multi-layer perceptron (MLP)** that uses the features only, so the difference between the two shows how much the links help.

**Why links should help.** Pages mostly like pages of their own type: 88.5% of links join two pages of the same type (this is called *homophily*). A politician's page is mostly liked by other politicians, so its neighbours are strong evidence of its type, even when its own description is vague.

**Headline result** (test set, mean of 3 runs): the GCN reaches **94.4% accuracy** and the MLP **81.5%**.

**The question beyond accuracy.** In practice, a model's predictions would be accepted automatically only when it is confident, with uncertain pages sent to a human. That only works if the model's confidence can be trusted. The links are also a weakness: for a page whose neighbours are mostly of a *different* type, the GCN is pulled towards the wrong answer, and it may be confidently wrong. The second half of this report tests whether that happens and recommends a rule for when to trust the model.

## How it works

![How the GCN classifies a page](figures/gcn_diagram.png)

*Figure 1. (a) In each layer, page A combines its own features with those of the pages it is linked to; here three of its four neighbours are politicians, which pulls A towards "politician". (b) Four such layers are stacked, so information travels up to four links; the final scores become probabilities, and the highest one is the model's confidence. The probabilities shown are illustrative.*

**The MLP (baseline)** looks at each page on its own. Its input is the page's 128 features, which it passes through layers. Each layer multiplies by a learned weight matrix and then applies ReLU (negative values set to zero). The last layer outputs 4 scores, one per type.

**The GCN** adds one step to every layer: *before* the multiplication, each page replaces its features with a weighted average of its own features and its neighbours' features. In matrix form, one layer is

$$H^{(l+1)} = \text{ReLU}\left(\hat{A}\, H^{(l)}\, W^{(l)}\right), \qquad \hat{A} = D^{-1/2}(A + I)\,D^{-1/2}$$

| Symbol | Meaning |
|---|---|
| $H^{(l)}$ | the features of every page entering layer $l$ (at the start, the 128 input features) |
| $A$ | the link table: $A_{ij} = 1$ if pages $i$ and $j$ are linked |
| $A + I$ | the links plus a link from each page to itself, so a page keeps its own features in the average |
| $D$ | the number of links of each page (counting the self-link) |
| $\hat{A}$ | the "averaging" matrix: the link between pages $i$ and $j$ gets weight $1/\sqrt{d_i d_j}$, so a page with hundreds of links does not drown out everyone it touches |
| $W^{(l)}$ | the layer's learned weights, shared by all pages |

Each layer reaches one link further. After 4 layers, a page's representation contains information from pages up to 4 links away. The MLP is exactly this network with $\hat{A}$ replaced by the identity matrix, i.e. with no averaging, which is why it is a fair baseline.

**From scores to confidence.** The 4 final scores are turned into probabilities with softmax. The predicted type is the one with the highest probability, and that probability is the model's **confidence**.

**Training.** The whole graph goes through the model at once, but the loss (cross-entropy) is computed only on the 60% of pages whose labels the model may see (the training set). The other pages still contribute their features and links, but not their labels. This is called *semi-supervised* learning. A further 20% (the validation set) is used to stop training at the epoch with the lowest validation loss, and the last 20% (the test set) is used once, at the end, to report results.

## Setup

Tested on a Windows 11 laptop **CPU only** (no GPU needed) with:

| Package | Version |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.14.1 (CPU build) |
| PyTorch Geometric | 2.8.1 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.4 |
| Matplotlib | 3.11.2 |

Install into a fresh virtual environment:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.8.1 scikit-learn==1.9.1 numpy==2.5.4 matplotlib==3.11.2
```

**Data.** The dataset is downloaded automatically on first run from the [Gayanku repository](https://github.com/gayanku/facebook_page-page_dataset) into `data/`, which is git-ignored and never committed. It is loaded with `allow_pickle=False`, so the file cannot run code when it is opened. The page-name array, which would need pickle, is skipped because it is not used.

## Usage

Run all commands from this folder (`recognition/GCN_s5042207/`).

| Command | What it does | Time (laptop CPU) |
|---|---|---|
| `python dataset.py` | Downloads the data if needed and prints the cleaning report, the data audit and the class counts in each split | a few seconds |
| `python train.py --model mlp` | Trains the 3-layer MLP baseline with seeds 0, 1 and 2 | about 1 min (≈ 20 s per seed) |
| `python train.py --model gcn` | Trains the 4-layer GCN with seeds 0, 1 and 2 | about 15 min (≈ 5 min per seed) |

Each training run saves:

| File | Contents | In git? |
|---|---|---|
| `checkpoints/{model}_seed{n}.pt` | the trained weights (best validation epoch) | no (ignored) |
| `figures/{model}_seed{n}_curves.png` | training and validation loss/accuracy curves | yes |
| `results/{model}_results.json` | settings, per-seed scores and mean ± std | yes |

Every setting can be changed from the command line, e.g. `python train.py --model gcn --layers 2 --seeds 0`. Run `python train.py --help` for the full list (layers, hidden size, dropout, learning rate, weight decay, maximum epochs, early-stopping patience, split seed).

**Reproducibility.** The train/validation/test split uses a fixed seed (42) and is identical for every model and run. Each training seed fixes the weight initialisation and dropout. Repeating the MLP runs on the same machine reproduced the saved results exactly; small differences are possible on other hardware or library versions.

## Feasibility Review

*Status as of 27 October 2026.*

### 1. User need, scope and acceptance criteria

**User:** a trust-and-safety team at a social platform that must sort verified pages into four categories (politician, government organisation, TV show, company), e.g. to apply political-advertising rules.
**Prototype purpose:** classify pages automatically from their descriptions and their mutual-like links, and send pages the model is unsure about to a human reviewer instead of guessing.
**Dilemma addressed:** does the graph model's confidence actually track correctness, or is it confidently wrong on ambiguous pages, such as pages whose neighbours belong to a different category?

Acceptance criteria (test split, mean over 3 seeds; set before the main experiments, after an initial single-seed smoke test: MLP 0.797, GCN 0.922 test accuracy):

1. **Accuracy:** GCN test macro-F1 ≥ 0.85 and higher than the MLP baseline on identical splits.
2. **Calibration:** after temperature scaling fitted on the validation set only, GCN expected calibration error (ECE) ≤ 0.05.
3. **Triage:** a confidence threshold chosen on validation automates ≥ 80% of test pages at ≥ 97% accuracy, with the rest routed to humans.
4. **Leakage:** train/val/test masks are disjoint (asserted in code), and test labels are never used for loss, early stopping, tuning or calibration.
5. **Resources:** training both models takes < 5 min on a laptop CPU with < 2 GB peak memory, and full-graph inference takes < 1 s.

### 2. Model choice and course concepts

- **Data:** Facebook Large Page-Page Network [Rozemberczki et al., 2021], modified PyG version by Gayanku: 22,470 pages, 128-dim features, 4 classes. Transductive semi-supervised setting with a stratified, fixed-seed train/val/test split.
- **Main model:** multi-layer GCN [Kipf & Welling, 2017], **Normal difficulty**. Each layer averages a page's features with its neighbours' and then applies a learned linear layer, so *k* layers see *k* hops. Starting from the standard 2 layers, a validation-only grid search (layers × weight decay × dropout × hidden size, 36 combinations, top settings confirmed over 3 seeds) selected **4 layers**. Accuracy only starts to drop at 8 layers (over-smoothing).
- **Baseline:** MLP with the same hidden size, dropout, weight decay and training budget, tuned by the same search (**3 layers**). It is the GCN without neighbour averaging, so any difference can be attributed to the graph. Label propagation (graph only, no features) is a second reference.
- **Concepts:** message passing, homophily [Zhu et al., 2020], calibration and temperature scaling [Guo et al., 2017], and GNN under-confidence [Wang et al., 2021].

### 3. Preliminary feasibility evidence

Data audit produced by `dataset.py`:

| Property | Value |
|---|---|
| Nodes / undirected edges | 22,470 / 170,823 (358 self-loops removed), no isolated nodes |
| Classes | tvshow 6,495 · government 6,880 · company 5,768 · politician 3,327 (14.8%, minority) |
| Degree | min 1 · median 7 · mean 15.2 · max 709 (heavy-tailed) |
| Edge homophily | 0.885 (class-insensitive 0.819) |

The high homophily suggests neighbour averaging should help overall, while the ~11.5% of cross-class edges are where the confidence dilemma should appear. The dataset author describes the file as a drop-in replacement for PyG's `FacebookPagePage`, but its array names differ, so the PyG class fails to read it; a custom loader was written instead.

**Baseline comparison** (`train.py`, 3 seeds, early stopping on validation loss, laptop CPU):

| Model | Parameters | Val accuracy | Test accuracy | Test macro-F1 |
|---|---|---|---|---|
| MLP, 3 layers (features only) | 12,676 | 0.814 ± 0.001 | 0.815 ± 0.002 | 0.798 ± 0.001 |
| GCN, 4 layers (features + graph) | 16,836 | 0.947 ± 0.002 | 0.944 ± 0.003 | 0.941 ± 0.003 |

The graph adds about 13 accuracy points, far beyond seed-to-seed variation, so the GCN is clearly viable (criterion 1's accuracy target is already met). Validation and test agree closely, so the model selection did not overfit the validation set.

**First look at the dilemma (validation only).** On pages where ≤ 25% of neighbours share the page's type (115 validation pages), the GCN is *worse* than the MLP (31% vs 50% accuracy) yet averages 84% confidence: exactly the confidently-wrong failure the investigation targets. A plain confidence threshold of 0.9 automates 83% of validation pages at 97.9% accuracy, so criterion 3 looks achievable.

**Resources:** one 4-layer GCN run takes about 5 minutes on a laptop CPU (the 2-layer version took about 2), so criterion 5 is borderline. Memory and inference time will be profiled.

### 4. Risks, budget and fallback

| Risk | Mitigation / fallback |
|---|---|
| Tight timeline (deadline 30 Oct) | Core analyses first (calibration, reject rule, failure autopsy). GAT was tried on validation and did not help, so it is reported briefly rather than developed further. |
| Confident errors are invisible to the model itself | Label-free warning signals (MLP–GCN disagreement, predicted-neighbour agreement) were tested on validation and added nothing beyond a higher threshold. Report this honestly as a residual risk that needs human review. |
| GCN training time exceeds the resource target | Cache the graph normalisation (done) and evaluate once per epoch; otherwise report the trade-off against the accuracy gain. |
| Noisy calibration estimates | Average over 3 seeds and report standard deviations. |

**Compute budget:** about 6 CPU-hours so far on a laptop, mostly an overnight hyperparameter search; no GPU needed (Colab as backup).
**Next experiment:** test-set evaluation of the final models: per-class metrics and AUROC, ECE and reliability diagrams before and after temperature scaling (fitted on validation), the confidence-threshold rule, and the local-homophily analysis.

### References

- T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," ICLR, 2017.
- B. Rozemberczki, C. Allen and R. Sarkar, "Multi-scale attributed node embedding," Journal of Complex Networks, 2021.
- C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On calibration of modern neural networks," ICML, 2017.
- J. Zhu et al., "Beyond homophily in graph neural networks: current limitations and effective designs," NeurIPS, 2020.
- X. Wang, H. Liu, C. Shi and C. Yang, "Be confident! Towards trustworthy graph neural networks via confidence calibration," NeurIPS, 2021.
