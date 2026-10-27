# Multi-layer GCN for Node Classification on the Facebook Page-Page Network

*COMP3710 Pattern Recognition Report — s5042207*

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
