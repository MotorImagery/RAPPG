## RAPPG: Relation-Aware Progressive Pseudo-label Generation (implementation)

This repository contains the code used for the paper
"RAPPG: Relation-Aware Progressive Pseudo-label Generation for Cross-subject MI-EEG Recognition."

The implementation is provided as reference for readers and researchers who want to understand or reproduce the method. It is an experimental, research-oriented codebase; a fully cleaned and reorganized release (a production-ready or well-packaged version) may be published in the future.

## Files

- `train.py` — Main training / evaluation script for BCI datasets (e.g., BCI Competition IV 2a / 2b) using ATCNet + the RAPPG pseudo-labeling flow.
- `train_hgd.py` — Training / evaluation script for the HGD dataset.
- `adaptation.py` — Fine-tuning / adaptation script that applies progressive pseudo-labels to the target domain.
- `atcnet_model.py` — ATCNet model implementation (feature extractor + classifier head).
- `ot_pseudolabel.py` — Implementation of the relation-aware progressive pseudo-label generation (OT-based components and helpers).
- `eegnet_model.py` — Referenced by several scripts but not included in this repository; provide your own EEGNet implementation or remove these imports if not needed.
- `datasets/` — Expected location for dataset `.mat` files used by the scripts.
- `ckpts/` — Directory used to save trained model checkpoints.

## Paper

RAPPG: Relation-Aware Progressive Pseudo-label Generation for Cross-subject MI-EEG Recognition


## Dependencies

The code imports the following third-party packages (please install appropriate versions for your environment):

- Python 3.7+
- numpy
- scipy
- torch (PyTorch)
- einops
- pot (Python Optimal Transport; imported as `ot` in the code)

Standard library modules used include `time` and `warnings`.

## Known issues / TODO

- The codebase is organized as experimental research code. Paths and interfaces are currently hard-coded in places and may be refactored in the future.



