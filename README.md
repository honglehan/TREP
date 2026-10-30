# Prioritizing the Rare Hits: A Top-Focused Ranking Framework for Extrapolating Protein Fitness

This repository provides two top-focused ranking objectives for protein fitness prediction on the Amylase, IRED, NucB, and TrpB two-to-many benchmarks. Models learn from precomputed protein language model embeddings to prioritize high-fitness variants.

## Contents

| Directory | Description |
|---|---|
| `pairwise_top_focus_ranking/` | Method based on pairwise ranking |
| `listwise_top_focus_ranking/` | Method based on listwise ranking |

## Installation

Use Python 3.10 or later and install a PyTorch build compatible with your CUDA environment. Install the remaining dependencies for the implementation you intend to run:

```bash
pip install -r requirements.txt
```
## Data preparation

We use the Amylase, IRED, NucB, and TrpB datasets from the FLIP2 two-to-many benchmark, following its provided training, validation, and test splits. Protein sequence embeddings are extracted using ESM2-650M and used as inputs to both the pairwise and listwise ranking methods.
