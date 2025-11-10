# Usage Guide

This guide explains how to use the path-embedding classifier for training and evaluation.

## Installation

Install the project and its dependencies:

```bash
cd path_embedding
uv sync --group dev
```

## Command Line Interface

### Training the Classifier

The main command is `train`, which orchestrates the entire training pipeline:

```bash
uv run path-embedding train \
  --data data/drugmechdb.yaml \
  --output models/my_classifier.pkl \
  --api-key-path /path/to/openai.key \
  --test-size 0.2 \
  --max-paths-per-indication 10 \
  --random-seed 42
```

#### Options

- `--data` (required): Path to DrugMechDB YAML file containing drug-disease mechanistic paths
- `--output` (required): Path where to save the trained model (.pkl)
- `--api-key-path`: Path to file containing OpenAI API key (default: `/Users/jtr4v/openai.key.another`)
- `--test-size`: Fraction of data for test set (default: 0.2)
- `--max-paths-per-indication`: Maximum paths to extract per indication (default: 10)
- `--random-seed`: Random seed for reproducibility (default: 42)

### Help

Get detailed help for the train command:

```bash
uv run path-embedding train --help
```

## Training Pipeline

The `train` command executes the following steps:

1. **Data Loading**: Loads DrugMechDB YAML file with multigraph representations
2. **Path Extraction**: Extracts all simple paths from drug to disease nodes
3. **Negative Sampling**: Generates negative examples via cross-disease shuffling
4. **Train/Test Split**: Splits at indication level to prevent data leakage
5. **Embedding Generation**: Converts paths to text and generates OpenAI embeddings
6. **Model Training**: Trains Random Forest classifier on embeddings
7. **Evaluation**: Evaluates on test set with standard metrics
8. **Model Saving**: Saves trained model to disk

## Example Workflow

### Step 1: Prepare Data

Place your DrugMechDB YAML file in the `data/` directory:

```bash
# Copy your data file
cp /path/to/drugmechdb.yaml data/
```

### Step 2: Set Up API Key

Create a file with your OpenAI API key:

```bash
echo "sk-your-api-key-here" > ~/.openai.key
chmod 600 ~/.openai.key
```

### Step 3: Train Model

Run the training pipeline:

```bash
uv run path-embedding train \
  --data data/drugmechdb.yaml \
  --output models/classifier.pkl \
  --api-key-path ~/.openai.key
```

### Step 4: Check Results

The training output will show:

```
Loading DrugMechDB data...
Loaded 50 indications
Extracting paths from multigraphs...
Extracted 150 positive paths
Generating negative examples...
Generated 150 negative paths
...
Training Random Forest classifier...
Evaluating on test set...

=== Classification Report ===
              precision    recall  f1-score   support
   Implausible       0.85      0.82      0.84        30
     Plausible       0.88      0.90      0.89        30
...

=== Metrics Summary ===
accuracy: 0.8667
precision: 0.8667
recall: 0.8667
f1: 0.8667
roc_auc: 0.9289
```

## Testing

Run the test suite:

```bash
# All tests with type checking and formatting
uv run pytest tests/ -v
```

Run specific test categories:

```bash
# Data loading tests
uv run pytest tests/test_drugmechdb.py -v

# Classifier tests
uv run pytest tests/test_classifier.py -v

# Evaluation tests
uv run pytest tests/test_evaluation.py -v

# CLI integration tests
uv run pytest tests/test_cli.py -v
```

Include doctests:

```bash
uv run pytest tests/ --doctest-modules -v
```

## Type Checking

Run mypy for type checking:

```bash
uv run mypy src tests
```

## Code Formatting

Check code style with ruff:

```bash
uv run ruff check .
```

## Architecture Overview

### Core Components

- **Data Models** (`src/path_embedding/datamodel/types.py`): Node, Edge, Path data structures
- **Data Loading** (`src/path_embedding/data/`): DrugMechDB and KGX loaders
- **Path Extraction** (`src/path_embedding/utils/path_extraction.py`): NetworkX multigraph operations
- **Embedding** (`src/path_embedding/embedding/`): Text conversion and OpenAI API integration
- **Classifier** (`src/path_embedding/model/`): Training, evaluation, and data splitting
- **CLI** (`src/path_embedding/cli.py`): Command-line interface

### Data Flow

```
DrugMechDB YAML
    ↓
Load Indications
    ↓
Build MultiDiGraphs
    ↓
Extract Paths
    ↓
Generate Negative Paths
    ↓
Convert to Text
    ↓
Generate Embeddings (OpenAI)
    ↓
Train/Test Split (by Indication)
    ↓
Train Random Forest
    ↓
Evaluate
    ↓
Save Model
```

## Performance Considerations

### API Costs

The training pipeline makes real API calls to OpenAI for embedding generation. Costs scale with:
- Number of paths
- Embedding model (text-embedding-3-small is default)

To minimize costs during development:
- Use small datasets or sample data
- Set `--max-paths-per-indication 1` to extract only one path per drug-disease pair
- Use `tests/data/sample_drugmechdb.yaml` (2 indications)

### Computation Time

Training time depends on:
- Dataset size (number of paths)
- API response times
- Random Forest hyperparameters

Typical training with 100 paths takes 2-5 minutes including API calls.

## Troubleshooting

### API Key Not Found

If you get an API key error, verify the file path and permissions:

```bash
cat /path/to/openai.key
chmod 600 /path/to/openai.key
```

### Missing Data File

Ensure the data file exists and is valid YAML:

```bash
ls -la data/drugmechdb.yaml
python -c "import yaml; yaml.safe_load(open('data/drugmechdb.yaml'))"
```

### Test Failures

Check that all dependencies are installed:

```bash
uv sync --group dev
```

Run tests with verbose output:

```bash
uv run pytest tests/ -vv -s
```

## Next Steps

- Experiment with different classifier hyperparameters
- Try alternative embeddings or classifiers
- Load data from KGX format (future implementation)
- Evaluate on larger datasets
- Deploy model for production use
