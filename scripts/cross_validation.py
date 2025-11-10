"""Cross-validation script for path embedding classifier.

This script performs 5-fold cross-validation to estimate performance with confidence intervals.
"""
import sys
import numpy as np
from sklearn.model_selection import KFold
from path_embedding.data.drugmechdb import load_drugmechdb
from path_embedding.utils.path_extraction import build_multigraph, extract_paths
from path_embedding.data.negative_sampling import generate_negatives
from path_embedding.embedding.openai_embedder import load_api_key, embed_paths
from path_embedding.model.classifier import train_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def main():
    data_path = "data/indication_paths.yaml"
    api_key_path = "/Users/jtr4v/openai.key.another"
    max_paths_per_indication = 10
    n_splits = 5
    random_seed = 42

    print("Loading DrugMechDB data...")
    indications = load_drugmechdb(data_path)
    print(f"Loaded {len(indications)} indications")

    print("Extracting paths from multigraphs...")
    # Group paths by indication
    indication_paths = {}
    skipped_count = 0

    for indication in indications:
        indication_id = indication["graph"]["_id"]
        graph = build_multigraph(indication)
        try:
            paths = extract_paths(graph, indication_id, max_paths=max_paths_per_indication)
            indication_paths[indication_id] = paths
        except ValueError as e:
            skipped_count += 1
            print(f"Warning: Skipping indication {indication_id}: {e}", file=sys.stderr)

    print(f"Extracted paths from {len(indication_paths)} indications (skipped {skipped_count})")

    # Convert to lists for cross-validation
    indication_ids = list(indication_paths.keys())

    print(f"\nPerforming {n_splits}-fold cross-validation...")
    print("=" * 70)

    # Store results
    fold_results = []

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    api_key = load_api_key(api_key_path)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(indication_ids), 1):
        print(f"\nFold {fold_idx}/{n_splits}")
        print("-" * 70)

        # Split indications
        train_indication_ids = [indication_ids[i] for i in train_idx]
        test_indication_ids = [indication_ids[i] for i in test_idx]

        # Gather paths
        train_pos = []
        for ind_id in train_indication_ids:
            train_pos.extend(indication_paths[ind_id])

        test_pos = []
        for ind_id in test_indication_ids:
            test_pos.extend(indication_paths[ind_id])

        print(f"Train: {len(train_pos)} positive paths from {len(train_indication_ids)} indications")
        print(f"Test: {len(test_pos)} positive paths from {len(test_indication_ids)} indications")

        # Generate negatives separately (NO LEAKAGE)
        print("Generating negatives...")
        train_neg = generate_negatives(train_pos)
        test_neg = generate_negatives(test_pos)

        # Combine
        train_paths = train_pos + train_neg
        test_paths = test_pos + test_neg

        train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))
        test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))

        # Generate embeddings
        print("Generating embeddings...")
        train_embeddings = embed_paths(train_paths, api_key)
        test_embeddings = embed_paths(test_paths, api_key)

        # Train and evaluate
        print("Training...")
        model = train_classifier(train_embeddings, train_labels, random_state=random_seed)

        print("Evaluating...")
        predictions = model.predict(test_embeddings)
        probabilities = model.predict_proba(test_embeddings)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        roc_auc = roc_auc_score(test_labels, probabilities)

        fold_results.append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })

        print(f"Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for metric in metrics:
        values = [r[metric] for r in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        ci_95 = 1.96 * std  # 95% confidence interval

        print(f"\n{metric.upper()}:")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std:  {std:.4f}")
        print(f"  95% CI: [{mean - ci_95:.4f}, {mean + ci_95:.4f}]")
        print(f"  Individual folds: {[f'{v:.4f}' for v in values]}")


if __name__ == "__main__":
    main()
