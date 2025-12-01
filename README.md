# Molecular Informatics Pipeline

This repository provides an end-to-end pipeline for downloading molecular data from the PubChem PUG REST API, featurizing molecules with RDKit, training machine learning models, and evaluating predictions on new molecules. The project is designed for reproducible cheminformatics experiments with clear data flow and logging.

## Repository Structure

```
project-root/
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_download.py
│   ├── featurize.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   └── exploratory.ipynb
├── models/
└── reports/
    └── model_performance.md
```

## Installation

1. Create and activate a Python environment (Python 3.9+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

RDKit is provided via the `rdkit-pypi` wheel for convenience.

## Configuration

All configurable options live in `config/settings.yaml`, including:
- PubChem API settings
- Default molecule identifiers for data acquisition and evaluation
- Fingerprint/descriptors configuration
- Model hyperparameters

Adjust the YAML file to suit your experiment before running the pipeline.

## Usage

1. **Download data**

```bash
python -m src.data_download --config config/settings.yaml
```

2. **Featurize molecules**

```bash
python -m src.featurize --config config/settings.yaml
```

3. **Train models**

```bash
python -m src.train --config config/settings.yaml
```

4. **Evaluate on new molecules**

```bash
python -m src.evaluate --config config/settings.yaml
```

Each stage logs progress to the console and writes outputs to the respective `data/`, `models/`, and `reports/` folders.

## Data Flow Diagram

```
[config/settings.yaml]
          |
          v
[data_download.py] --(PubChem PUG REST)--> data/raw/compounds.csv
          |
          v
[featurize.py] --> data/processed/processed.csv
          |
          v
[train.py] --> models/{ridge.pkl,random_forest.pkl} + reports/feature_importance.png
          |
          v
[evaluate.py] --> reports/model_performance.md
```

## Limitations
- PubChem rate limits may slow down large batch downloads.
- Synthetic regression targets are generated if none are provided; results should not be considered domain-validated without real labels.
- Fingerprints/descriptors are computed with default RDKit settings; explore alternatives for specialized tasks.

## Citation
If you use this pipeline, please cite PubChem and RDKit appropriately.
