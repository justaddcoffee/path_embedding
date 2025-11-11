# Hard Negatives Experiment Design

**Date:** 2025-11-11
**Purpose:** Create more challenging negative examples by replacing only one node instead of shuffling all intermediate nodes

## Background

Current negative sampling (`generate_negatives()`) replaces **all intermediate nodes** with nodes from different disease contexts. This creates very obviously wrong paths.

This experiment tests whether replacing just **one node** creates harder negatives that force the model to learn more subtle distinctions.

## Design Decisions

### 1. Node Selection Strategy
- **Random intermediate node** - Select one random non-Drug/non-Disease node
- Simple, unbiased approach
- Creates variety in what gets changed

### 2. Collision Avoidance
- **Check against extracted positive paths** - Compare generated path text against all known positives
- Fast O(1) lookup using set
- Retry with different random node if collision detected (max 10 attempts)
- Fall back to full shuffle if all retries fail

### 3. Replacement Strategy
- **Sample from different disease context** - Replace with node of same type from different disease
- Reuses existing `build_node_inventory()` structure
- Different disease context helps ensure implausibility
- Single-node change makes it "hard" regardless

### 4. Code Integration
- **New function alongside existing** - Create `generate_hard_negatives()`
- Keeps both approaches available for comparison
- Separate experiment notebook to evaluate both

## Implementation

### New Function: `generate_hard_negative_path()`

```python
def generate_hard_negative_path(
    positive_path: Path,
    node_inventory: Dict[str, Dict[str, List[Node]]],
    all_positive_texts: set[str],
    max_retries: int = 10
) -> Path
```

**Algorithm:**
1. Get all intermediate nodes (exclude Drug/Disease)
2. If no intermediate nodes, fall back to full shuffle
3. Randomly select one intermediate node position
4. For up to `max_retries` attempts:
   - Replace selected node with random node of same type from different disease
   - Convert path to text using `path_to_text()`
   - Check if text in `all_positive_texts`
   - If not found: return this negative path
5. If all retries fail: fall back to `generate_negative_path()` (full shuffle)

**Why retry limit of 10?**
- Single-node changes are unlikely to collide with real paths
- 10 retries gives high probability of finding non-colliding path
- Fallback ensures we always get a negative example

### New Function: `generate_hard_negatives()`

```python
def generate_hard_negatives(positive_paths: List[Path]) -> List[Path]
```

**Algorithm:**
1. Build node inventory using existing `build_node_inventory()`
2. Pre-compute set of all positive path texts for collision detection
3. For each positive path, generate one hard negative
4. Returns list of negative paths (maintains 1:1 ratio)

## Experiment Design

### Notebook: `notebooks/hard_negatives_experiment.ipynb`

**Structure:**

1. **Setup & Data Loading**
   - Load DrugMechDB data
   - Extract paths from multigraphs
   - Split train/test by indication

2. **Experiment 1: Standard Negatives (Baseline)**
   - Generate negatives with `generate_negatives()` (shuffle all nodes)
   - Train Random Forest + OpenAI embeddings
   - Train Logistic Regression + TF-IDF
   - Record metrics (accuracy, precision, recall, F1, ROC-AUC)

3. **Experiment 2: Hard Negatives**
   - Generate negatives with `generate_hard_negatives()` (replace one node)
   - Train same two models
   - Record same metrics

4. **Comparison & Analysis**
   - Side-by-side metrics comparison table
   - Visualizations showing performance differences
   - Error analysis: which negatives are harder to classify?
   - Example comparisons of standard vs hard negatives
   - Statistical significance testing

**Expected Outcomes:**

- **Hard negatives hypothesis:** Lower accuracy but more robust learning
- **Model discrimination:** May reveal which model better captures mechanistic understanding
- **Error patterns:** Hard negatives should have lower confidence scores

## Testing

Add tests to `tests/test_negative_sampling.py`:

1. **Test single-node difference:**
   - Verify hard negative differs from positive by exactly one intermediate node
   - Drug and Disease nodes remain unchanged

2. **Test collision detection:**
   - Verify function rejects paths that match existing positives
   - Verify retry mechanism works

3. **Test fallback behavior:**
   - When retries exhausted, falls back to full shuffle
   - When no intermediate nodes exist, falls back to full shuffle

4. **Test node type preservation:**
   - Replacement node has same label as original node

## Success Criteria

1. Function generates valid hard negatives (one node different)
2. No false negatives (real paths labeled as negative)
3. Experiment notebook runs end-to-end
4. Results clearly show difficulty difference between standard and hard negatives
5. All tests pass

## Files to Create/Modify

**New files:**
- `notebooks/hard_negatives_experiment.ipynb`

**Modified files:**
- `src/path_embedding/data/negative_sampling.py` - Add `generate_hard_negative_path()` and `generate_hard_negatives()`
- `tests/test_negative_sampling.py` - Add tests for new functions

## Open Questions

None - design is complete and validated.
