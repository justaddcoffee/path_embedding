# Path Embedding Classifier Design

**Date**: 2025-11-10
**Status**: Design Complete, Ready for Implementation

## Overview

Build a binary classifier that distinguishes biologically plausible from implausible mechanistic paths between drugs and diseases using embedding-based representations.

## Core Approach

1. Convert mechanistic paths to structured text representations
2. Generate embeddings using OpenAI's embedding API
3. Train a Random Forest classifier on the embeddings
4. Evaluate on held-out test data

## Data Source: DrugMechDB

DrugMechDB provides curated, biologically plausible drug-disease mechanistic paths in YAML format.

**Data Structure**:
- Each entry (identified by `_id`) is a multigraph containing:
  - Nodes with `id`, `label` (type like Drug, Gene, Protein, Disease), and `name`
  - Edges (links) with `key` (relationship type), `source`, and `target`
- Source: https://raw.githubusercontent.com/SuLab/DrugMechDB/refs/heads/main/indication_paths.yaml

**Path Extraction Strategy**:
1. Load each DrugMechDB multigraph using NetworkX
2. Extract all simple paths from drug node to disease node using `nx.all_simple_paths()`
3. **Critical Design Decision**: Each extracted path = one separate training example (labeled as plausible)
4. If an indication yields >10 paths, randomly sample 10 to prevent dataset imbalance
5. This creates our positive training examples

## Negative Example Generation

**Strategy**: Cross-disease shuffling with type matching

For each positive path, create a negative example by:
1. Keep the same drug (start) and disease (end) nodes
2. For each intermediate position, maintain the node type (e.g., Gene, Protein, Pathway)
3. Substitute a random node of that type from a **different disease context**
4. This creates challenging negatives that preserve structural patterns but break biological plausibility

**Rationale**: Simple random shuffling could accidentally create plausible paths. Cross-disease shuffling reduces this risk while creating realistic-looking but implausible examples.

**Alternative Strategies** (noted for future exploration):
- Random walks through combined node/edge pool
- Stratified random sampling
- Combination of multiple strategies

## Text Representation

**Format**: Structured template with clear delimiters

**Example**:
```
Drug: aspirin | decreases activity of | Protein: COX2 | causes | Disease: inflammation
```

**Specification**:
- Alternate between nodes and edges when traversing the path
- Nodes: `{label}: {name}` (e.g., "Protein: BCR/ABL")
- Edges: relationship type from the `key` field
- Delimiter: ` | ` (space-pipe-space) between elements
- When multiple edges exist between the same node pair (multigraph), use bundle notation: `[edge1|edge2]`

**Rationale**:
- Clear separation between path elements for the embedding model
- Node type context alongside names
- Readable and machine-parseable
- Preserves all mechanistic information

## Future Enhancement: Node Context Enrichment

**Not implemented initially** - start simple, add if needed:
- Include one-hop neighbors (subclass-of, part-of relationships) for additional context
- Would require external ontology lookups (MESH, DrugBank, UniProt, etc.)
- **Important note**: Previous experiments showed significant improvement for diagnostic use cases when including one-hop neighbors
- Plan to add this in a future iteration if basic approach works

## Embedding Generation

- **Model**: OpenAI's text embedding API (text-embedding-3-small or similar)
- **API Key**: Load from `/Users/jtr4v/openai.key.another`
- **Process**: Each path text → single fixed-size embedding vector
- **Length handling**: No concerns initially - biological mechanism paths are typically short enough for token limits

## Classifier Training

**Classifier**: Random Forest (scikit-learn)
- Start with default hyperparameters for simplicity
- Future consideration: XGBoost if performance needs improvement

**Train/Test Split**:
- **80/20 split at the drug-disease indication level**
- Group all paths belonging to the same drug-disease pair together
- Split at the indication level (not individual path level) to prevent data leakage
- Ensures the model learns general mechanistic patterns rather than memorizing specific drug-disease combinations
- Use fixed random seed for reproducibility

**Class Balance**:
- 1:1 ratio of positive to negative examples
- For every plausible path, generate one implausible path
- Both train and test sets maintain this balance

## Evaluation

**Metrics** (standard binary classification):
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Report all metrics on the held-out 20% test set.

**Output**:
- Trained model serialized to disk (pickle or joblib)
- Classification report and confusion matrix
- Performance metrics summary

## Future Extensions

### KGX Format Support
- Create stub/placeholder for KGX path ingestion
- KGX (Knowledge Graph Exchange) is a JSON/TSV format for knowledge graphs
- Will need adapter to convert KGX nodes/edges to the same internal path representation
- **Not implemented initially** - DrugMechDB first, validate approach works

### Alternative Classifiers
- XGBoost
- Neural network approaches
- Ensemble methods

## Project Structure

```
path-embedding/
├── src/path_embedding/
│   ├── cli.py                    # Typer CLI interface
│   ├── data/
│   │   ├── drugmechdb.py        # DrugMechDB loading and path extraction
│   │   ├── kgx.py               # KGX stub (future)
│   │   └── negative_sampling.py # Cross-disease shuffling logic
│   ├── embedding/
│   │   ├── text_formatter.py    # Path → structured text conversion
│   │   └── openai_embedder.py   # OpenAI API integration
│   ├── model/
│   │   ├── classifier.py        # Random Forest training/evaluation
│   │   └── evaluation.py        # Metrics and reporting
│   └── utils/
│       └── path_extraction.py   # NetworkX path extraction utilities
├── tests/
│   ├── test_path_extraction.py
│   ├── test_text_formatter.py
│   ├── test_negative_sampling.py
│   └── test_integration.py
├── data/                         # Downloaded DrugMechDB YAML
└── models/                       # Saved trained models
```

## CLI Interface (Proposed)

```bash
# Train classifier on DrugMechDB
uv run path-embedding train --data data/indication_paths.yaml --output models/classifier.pkl

# Evaluate on test set
uv run path-embedding evaluate --model models/classifier.pkl --test-data data/test.pkl

# Predict plausibility of paths from KGX (future)
uv run path-embedding predict --model models/classifier.pkl --input paths.kgx.json
```

## Key Design Decisions Summary

1. **Each path = separate example**: Extract all simple paths from multigraphs, treat each as independent training example
2. **Cross-disease shuffling**: Generate negatives by swapping nodes across disease contexts while preserving types
3. **Structured text format**: Use template with delimiters for clear, parseable path representation
4. **Indication-level split**: Split train/test at drug-disease pair level to prevent leakage
5. **Start simple**: Random Forest, basic text representation, add complexity incrementally
6. **OpenAI embeddings**: Use commercial API for high-quality embeddings
7. **DrugMechDB first**: Validate approach before adding KGX support

## Implementation Priority

1. DrugMechDB data loading and path extraction (NetworkX)
2. Text formatting and OpenAI embedding generation
3. Negative example generation (cross-disease shuffling)
4. Train/test splitting (indication-level)
5. Random Forest training and evaluation
6. CLI interface for training and evaluation
7. KGX stub (minimal, for future work)
