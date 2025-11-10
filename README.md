
# path-embedding

A binary classifier that distinguishes biologically plausible from implausible mechanistic paths between drugs and diseases using embedding-based representations.

## Overview

This tool uses large language model embeddings to classify whether a given mechanistic path from a drug to a disease is biologically plausible. The classifier is trained on curated data from [DrugMechDB](https://sulab.github.io/DrugMechDB/), which contains validated drug mechanism of action pathways.

### How It Works

1. **Path Representation**: Mechanistic paths (sequences of biological entities and their relationships) are converted to structured text
2. **Embedding Generation**: Text representations are embedded using OpenAI's embedding API
3. **Classification**: A Random Forest classifier trained on these embeddings predicts path plausibility

### Use Cases

- Validate proposed drug mechanisms of action
- Filter candidate paths from knowledge graph traversals
- Prioritize mechanistic hypotheses for experimental validation
- Quality control for automated pathway extraction

## Quick Start

```bash
# Train classifier on DrugMechDB
uv run path-embedding train --data data/indication_paths.yaml --output models/classifier.pkl

# Evaluate on test set
uv run path-embedding evaluate --model models/classifier.pkl --test-data data/test.pkl
```

## Documentation Website

[https://justaddcoffee.github.io/path-embedding](https://justaddcoffee.github.io/path-embedding)

## Key Features

- **Embedding-based classification**: Leverages OpenAI embeddings to capture semantic meaning of biological pathways
- **Curated training data**: Trained on expert-validated mechanisms from DrugMechDB
- **Negative sampling**: Generates challenging negative examples using cross-disease shuffling with type matching
- **Indication-level splitting**: Prevents data leakage by splitting train/test at drug-disease pair level
- **Extensible design**: Ready for future integration with KGX format knowledge graphs

## Repository Structure

* [docs/](docs/) - mkdocs-managed documentation
  * [plans/](docs/plans/) - Design documents and implementation plans
* [project/](project/) - project files (these files are auto-generated, do not edit)
* [src/](src/) - source files (edit these)
  * [path_embedding](src/path_embedding)
    * `data/` - Data loading and path extraction (DrugMechDB, KGX stub)
    * `embedding/` - Text formatting and embedding generation
    * `model/` - Classifier training and evaluation
* [tests/](tests/) - Python tests
  * [data/](tests/data) - Example data
* [data/](data/) - Downloaded datasets (DrugMechDB YAML)
* [models/](models/) - Trained classifier models

## Methodology

See [Design Document](docs/plans/2025-11-10-path-embedding-classifier-design.md) for detailed design decisions and rationale.

### Training Data

- **Positive examples**: Curated mechanistic paths from DrugMechDB
- **Negative examples**: Generated via cross-disease shuffling with type matching
  - Preserves path structure and node types
  - Swaps intermediate nodes from different disease contexts
  - Creates challenging negatives that break biological plausibility

### Text Representation

Paths are converted to structured text format:
```
Drug: aspirin | decreases activity of | Protein: COX2 | causes | Disease: inflammation
```

### Evaluation

- 80/20 train/test split at drug-disease indication level
- Standard binary classification metrics: accuracy, precision, recall, F1-score, ROC-AUC
- Balanced classes (1:1 positive:negative ratio)

## Requirements

- Python 3.10+
- OpenAI API key (for embeddings)
- Dependencies managed via `uv`

## Developer Tools

There are several pre-defined command-recipes available.
They are written for the command runner [just](https://github.com/casey/just/). To list all pre-defined commands, run `just` or `just --list`.

Key commands:
- `just test` - Run all tests, type checking, and formatting checks
- `just pytest` - Run Python tests only
- `just mypy` - Run type checking
- `just format` - Run ruff linting/formatting checks

## Future Work

- **Node context enrichment**: Include one-hop neighbors (subclass-of, part-of) for richer representations
  - Previous experiments showed significant improvements for diagnostic use cases
- **KGX format support**: Apply trained classifier to arbitrary paths from knowledge graphs
- **Alternative classifiers**: XGBoost, neural networks, ensemble methods
- **Alternative negative sampling**: Random walks, stratified sampling, combinations

## Credits

This project uses the template [monarch-project-copier](https://github.com/monarch-initiative/monarch-project-copier)
