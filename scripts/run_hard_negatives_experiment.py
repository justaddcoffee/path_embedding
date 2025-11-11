"""Run hard negatives experiment and save results."""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from path_embedding.data.drugmechdb import load_drugmechdb
from path_embedding.utils.path_extraction import build_multigraph, extract_paths
from path_embedding.data.negative_sampling import generate_negatives, generate_hard_negatives
from path_embedding.embedding.openai_embedder import load_api_key, embed_paths
from path_embedding.embedding.text_formatter import path_to_text
from path_embedding.model.classifier import train_classifier
from path_embedding.model.data_split import split_by_indication
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configuration
DATA_PATH = 'data/indication_paths.yaml'
API_KEY_PATH = '/Users/jtr4v/openai.key.another'
MAX_PATHS_PER_INDICATION = 10
TEST_SIZE = 0.2
RANDOM_SEED = 42

print("="*70)
print("HARD NEGATIVES EXPERIMENT")
print("="*70)

# 1. Load and prepare data
print("\n1. Loading DrugMechDB data...")
indications = load_drugmechdb(DATA_PATH)
print(f"   Loaded {len(indications)} indications")

print("\n2. Extracting paths from multigraphs...")
all_positive_paths = []
skipped_count = 0

for indication in indications:
    graph = build_multigraph(indication)
    try:
        paths = extract_paths(graph, indication['graph']['_id'], max_paths=MAX_PATHS_PER_INDICATION)
        all_positive_paths.extend(paths)
    except ValueError:
        skipped_count += 1

print(f"   Extracted {len(all_positive_paths)} positive paths (skipped {skipped_count} invalid indications)")

print("\n3. Splitting train/test by indication...")
train_pos, test_pos = split_by_indication(all_positive_paths, test_size=TEST_SIZE, random_seed=RANDOM_SEED)
print(f"   Train: {len(train_pos)} positive")
print(f"   Test: {len(test_pos)} positive")

# Load API key
print("\n4. Loading API key...")
api_key = load_api_key(API_KEY_PATH)

# ============================================================================
# EXPERIMENT 1: STANDARD NEGATIVES
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 1: STANDARD NEGATIVES (SHUFFLE ALL NODES)")
print("="*70)

print("\n5. Generating standard negatives...")
train_neg_standard = generate_negatives(train_pos)
test_neg_standard = generate_negatives(test_pos)
print(f"   Train: {len(train_neg_standard)} negatives")
print(f"   Test: {len(test_neg_standard)} negatives")

train_paths_standard = train_pos + train_neg_standard
test_paths_standard = test_pos + test_neg_standard

train_labels_standard = np.array([1] * len(train_pos) + [0] * len(train_neg_standard))
test_labels_standard = np.array([1] * len(test_pos) + [0] * len(test_neg_standard))

print("\n6. Generating embeddings (standard negatives)...")
print("   This will take a few minutes...")
train_embeddings_standard = embed_paths(train_paths_standard, api_key)
print(f"   Train embeddings: {train_embeddings_standard.shape}")

test_embeddings_standard = embed_paths(test_paths_standard, api_key)
print(f"   Test embeddings: {test_embeddings_standard.shape}")

print("\n7. Training Random Forest (standard negatives)...")
rf_model_standard = train_classifier(train_embeddings_standard, train_labels_standard, random_state=RANDOM_SEED)

print("\n8. Evaluating (standard negatives)...")
rf_predictions_standard = rf_model_standard.predict(test_embeddings_standard)
rf_probabilities_standard = rf_model_standard.predict_proba(test_embeddings_standard)[:, 1]

standard_results = {
    'accuracy': float(accuracy_score(test_labels_standard, rf_predictions_standard)),
    'precision': float(precision_score(test_labels_standard, rf_predictions_standard)),
    'recall': float(recall_score(test_labels_standard, rf_predictions_standard)),
    'f1': float(f1_score(test_labels_standard, rf_predictions_standard)),
    'roc_auc': float(roc_auc_score(test_labels_standard, rf_probabilities_standard))
}

print("\n   Standard Negatives Results:")
for metric, value in standard_results.items():
    print(f"     {metric}: {value:.4f}")

# ============================================================================
# EXPERIMENT 2: HARD NEGATIVES
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 2: HARD NEGATIVES (REPLACE ONE NODE)")
print("="*70)

print("\n9. Generating hard negatives...")
train_neg_hard = generate_hard_negatives(train_pos)
test_neg_hard = generate_hard_negatives(test_pos)
print(f"   Train: {len(train_neg_hard)} negatives")
print(f"   Test: {len(test_neg_hard)} negatives")

train_paths_hard = train_pos + train_neg_hard
test_paths_hard = test_pos + test_neg_hard

train_labels_hard = np.array([1] * len(train_pos) + [0] * len(train_neg_hard))
test_labels_hard = np.array([1] * len(test_pos) + [0] * len(test_neg_hard))

print("\n10. Generating embeddings (hard negatives)...")
print("    This will take a few minutes...")
train_embeddings_hard = embed_paths(train_paths_hard, api_key)
print(f"    Train embeddings: {train_embeddings_hard.shape}")

test_embeddings_hard = embed_paths(test_paths_hard, api_key)
print(f"    Test embeddings: {test_embeddings_hard.shape}")

print("\n11. Training Random Forest (hard negatives)...")
rf_model_hard = train_classifier(train_embeddings_hard, train_labels_hard, random_state=RANDOM_SEED)

print("\n12. Evaluating (hard negatives)...")
rf_predictions_hard = rf_model_hard.predict(test_embeddings_hard)
rf_probabilities_hard = rf_model_hard.predict_proba(test_embeddings_hard)[:, 1]

hard_results = {
    'accuracy': float(accuracy_score(test_labels_hard, rf_predictions_hard)),
    'precision': float(precision_score(test_labels_hard, rf_predictions_hard)),
    'recall': float(recall_score(test_labels_hard, rf_predictions_hard)),
    'f1': float(f1_score(test_labels_hard, rf_predictions_hard)),
    'roc_auc': float(roc_auc_score(test_labels_hard, rf_probabilities_hard))
}

print("\n    Hard Negatives Results:")
for metric, value in hard_results.items():
    print(f"      {metric}: {value:.4f}")

# ============================================================================
# COMPARISON & ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("COMPARISON & ANALYSIS")
print("="*70)

comparison_df = pd.DataFrame({
    'Standard Negatives': standard_results,
    'Hard Negatives': hard_results
})
comparison_df['Difference'] = comparison_df['Standard Negatives'] - comparison_df['Hard Negatives']

print("\n" + comparison_df.round(4).to_string())
print("\nNote: Positive difference means standard negatives are easier to classify")

# Error Analysis
standard_errors = test_labels_standard != rf_predictions_standard
hard_errors = test_labels_hard != rf_predictions_hard

print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70)
print(f"\nStandard Negatives:")
print(f"  Total errors: {standard_errors.sum()} / {len(test_labels_standard)} ({standard_errors.sum()/len(test_labels_standard)*100:.2f}%)")

print(f"\nHard Negatives:")
print(f"  Total errors: {hard_errors.sum()} / {len(test_labels_hard)} ({hard_errors.sum()/len(test_labels_hard)*100:.2f}%)")

# Example comparison
print("\n" + "="*70)
print("EXAMPLE COMPARISON")
print("="*70)
example_idx = 0
positive_example = train_pos[example_idx]
standard_negative_example = train_neg_standard[example_idx]
hard_negative_example = train_neg_hard[example_idx]

print("\nPositive Path:")
print(path_to_text(positive_example)[:200] + "...")

print("\nStandard Negative (all nodes shuffled):")
print(path_to_text(standard_negative_example)[:200] + "...")

print("\nHard Negative (one node replaced):")
print(path_to_text(hard_negative_example)[:200] + "...")

# Count node differences
standard_diffs = sum(1 for i in range(len(positive_example.nodes))
                     if positive_example.nodes[i].id != standard_negative_example.nodes[i].id)
hard_diffs = sum(1 for i in range(len(positive_example.nodes))
                 if positive_example.nodes[i].id != hard_negative_example.nodes[i].id)

print(f"\nNode differences from positive:")
print(f"  Standard negative: {standard_diffs} nodes different")
print(f"  Hard negative: {hard_diffs} nodes different")

# Conclusions
print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)

acc_diff = standard_results['accuracy'] - hard_results['accuracy']

print("\n1. DIFFICULTY")
if acc_diff > 0.05:
    print(f"   Hard negatives are SIGNIFICANTLY harder to classify ({acc_diff:+.2%} accuracy drop)")
elif acc_diff > 0.02:
    print(f"   Hard negatives are MODERATELY harder to classify ({acc_diff:+.2%} accuracy drop)")
else:
    print(f"   Hard negatives have SIMILAR difficulty ({acc_diff:+.2%} accuracy change)")

print("\n2. PERFORMANCE")
print(f"   Standard Negatives: {standard_results['accuracy']:.2%} accuracy")
print(f"   Hard Negatives: {hard_results['accuracy']:.2%} accuracy")

print("\n3. RECOMMENDATION")
if hard_results['accuracy'] > 0.85:
    print("   Hard negatives still achieve good performance (>85% accuracy)")
    print("   Consider using for more robust model training")
elif hard_results['accuracy'] > 0.75:
    print("   Hard negatives achieve reasonable performance (75-85% accuracy)")
    print("   May be useful for specific applications requiring robustness")
else:
    print("   Hard negatives may be too difficult (<75% accuracy)")
    print("   Standard negatives recommended for this task")

# Save results
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

results = {
    'standard_negatives': standard_results,
    'hard_negatives': hard_results,
    'comparison': comparison_df.to_dict(),
    'config': {
        'data_path': DATA_PATH,
        'max_paths_per_indication': MAX_PATHS_PER_INDICATION,
        'test_size': TEST_SIZE,
        'random_seed': RANDOM_SEED,
        'total_positive_paths': len(all_positive_paths),
        'train_positive': len(train_pos),
        'test_positive': len(test_pos)
    }
}

output_file = output_dir / "hard_negatives_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nResults saved to: {output_file}")
print("\n" + "="*70)
print("EXPERIMENT COMPLETE!")
print("="*70)
