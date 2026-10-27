# Multi-layer GCN for Node Classification on the Facebook Page-Page Network

*COMP3710 Pattern Recognition Report — s5042207*

## Feasibility Review (draft)

### 1. User need, scope and acceptance criteria

**User:** a trust-and-safety team at a social platform that must sort verified pages into four categories (politician, government organisation, TV show, company), e.g. to apply political-advertising rules.
**Prototype purpose:** classify pages automatically from their descriptions and their mutual-like links, and send pages the model is unsure about to a human reviewer instead of guessing.
**Dilemma addressed:** does the graph model's confidence actually track correctness, or is it confidently wrong on ambiguous pages, such as pages whose neighbours belong to a different category?

Acceptance criteria (test split, mean over 3 seeds; set before the main experiments, informed by the smoke test in Section 3):

1. **Accuracy:** GCN test macro-F1 ≥ 0.85 and higher than the MLP baseline on identical splits.
2. **Calibration:** after temperature scaling fitted on the validation set only, GCN expected calibration error (ECE) ≤ 0.05.
3. **Triage:** a confidence threshold chosen on validation automates ≥ 80% of test pages at ≥ 97% accuracy, with the rest routed to humans.
4. **Leakage:** train/val/test masks are disjoint (asserted in code), and test labels are never used for loss, early stopping, tuning or calibration.
5. **Resources:** training both models takes < 5 min on a laptop CPU with < 2 GB peak memory, and full-graph inference takes < 1 s.

### 2. Model choice and course concepts

- **Data:** Facebook Large Page-Page Network [Rozemberczki et al., 2021], modified PyG version by Gayanku: 22,470 pages, 128-dim features, 4 classes. Transductive semi-supervised setting with a stratified, fixed-seed train/val/test split.
- **Main model:** 2-layer GCN [Kipf & Welling, 2017], **Normal difficulty**. Each layer averages a page's features with its neighbours' and then applies a learned linear layer, so two layers see friends-of-friends. Two layers avoid over-smoothing.
- **Baseline:** 2-layer MLP with the same hidden size, dropout and training budget. It is the GCN without neighbour averaging, so any difference can be attributed to the graph.
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

**Baseline smoke test** (single seed, 200 epochs, best epoch chosen by validation accuracy, Google Colab CPU):

| Model | Val accuracy | Test accuracy | Training time |
|---|---|---|---|
| MLP (features only) | 0.804 | 0.797 | 11.7 s |
| GCN (features + graph) | 0.922 | 0.922 | 104.1 s |

The graph adds about 12 accuracy points, so the GCN is clearly viable, and the remaining ~8% of GCN errors are the cases the confidence investigation will focus on. Both models were still improving slightly at epoch 200, so the full runs use up to 300 epochs with early stopping on validation loss. Cost is small for both models (< 2 min per run on a free CPU), so the resource criterion is already met.

### 4. Risks, budget and fallback

| Risk | Mitigation / fallback |
|---|---|
| Tight timeline (5 days) | MLP + GCN + calibration are the core; GAT and structural-feature analysis are optional extras dropped first. |
| GCN not better calibrated than the MLP | This is a valid finding for the dilemma, not a failure; report it with evidence. |
| Noisy calibration estimates | Average over ≥ 3 seeds and report standard deviations. |
| Dataset link disappears | Loader caches the file locally; a copy can be re-hosted. |

**Compute budget:** < 1 CPU-hour in total on a laptop, with no GPU needed (Colab as backup).
**Next experiment:** full MLP and GCN runs over 3 seeds with `train.py` (macro-F1, per-class metrics), then confidence calibration on the validation set.

### References

- T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," ICLR, 2017.
- B. Rozemberczki, C. Allen and R. Sarkar, "Multi-scale attributed node embedding," Journal of Complex Networks, 2021.
- C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On calibration of modern neural networks," ICML, 2017.
- J. Zhu et al., "Beyond homophily in graph neural networks: current limitations and effective designs," NeurIPS, 2020.
- X. Wang, H. Liu, C. Shi and C. Yang, "Be confident! Towards trustworthy graph neural networks via confidence calibration," NeurIPS, 2021.
