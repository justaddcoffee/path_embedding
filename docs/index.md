# path-embedding

Classifier that uses embeddings to find useful paths between drugs and disease

## Quick Links

- [Usage Guide](usage.md) - How to train and evaluate the classifier
- [Design Document](plans/2025-11-10-path-embedding-classifier-design.md) - Detailed architecture and design decisions
- [Implementation Plan](plans/2025-11-10-path-embedding-implementation.md) - Complete task breakdown

## Overview

This tool uses large language model embeddings (OpenAI) to classify whether mechanistic paths between drugs and diseases are biologically plausible.

### Key Features

- **Embedding-based classification**: Leverages OpenAI text-embedding-3-small to capture semantic meaning
- **Negative sampling**: Cross-disease shuffling with type matching creates challenging negatives
- **Indication-level split**: Prevents data leakage by splitting at drug-disease pair level
- **Comprehensive evaluation**: Standard metrics (accuracy, precision, recall, F1, ROC-AUC)
- **Extensible design**: Ready for KGX format and other knowledge graphs

### Training Pipeline

```
DrugMechDB YAML → Extract Paths → Generate Negatives → Convert to Text →
Embed (OpenAI) → Train/Test Split → Train Random Forest → Evaluate → Save Model
```

## Getting Started

```bash
# Install dependencies
uv sync --group dev

# Train classifier
uv run path-embedding train \
  --data data/drugmechdb.yaml \
  --output models/classifier.pkl \
  --api-key-path ~/.openai.key

# Run tests
uv run pytest tests/ -v
```

## Documentation

- Auto-generated [schema documentation](elements/index.md)
