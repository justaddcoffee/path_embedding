"""CLI interface for path-embedding."""

import typer
from typing_extensions import Annotated
from pathlib import Path
import pickle
import numpy as np

app = typer.Typer(help="path-embedding: Classifier that uses embeddings to find useful paths between drugs and disease")


@app.command()
def train(
    data: Annotated[str, typer.Option(help="Path to DrugMechDB YAML file")],
    output: Annotated[str, typer.Option(help="Path to save trained model (.pkl)")],
    api_key_path: Annotated[str, typer.Option(help="Path to OpenAI API key file")] = "/Users/jtr4v/openai.key.another",
    test_size: Annotated[float, typer.Option(help="Fraction for test set")] = 0.2,
    max_paths_per_indication: Annotated[int, typer.Option(help="Max paths to extract per indication")] = 10,
    random_seed: Annotated[int, typer.Option(help="Random seed")] = 42,
):
    """Train path embedding classifier on DrugMechDB data."""
    from path_embedding.data.drugmechdb import load_drugmechdb
    from path_embedding.utils.path_extraction import build_multigraph, extract_paths
    from path_embedding.data.negative_sampling import generate_negatives
    from path_embedding.model.data_split import split_by_indication
    from path_embedding.embedding.openai_embedder import load_api_key, embed_paths
    from path_embedding.model.classifier import train_classifier
    from path_embedding.model.evaluation import evaluate_classifier, print_evaluation_report

    typer.echo("Loading DrugMechDB data...")
    indications = load_drugmechdb(data)
    typer.echo(f"Loaded {len(indications)} indications")

    typer.echo("Extracting paths from multigraphs...")
    all_positive_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"], max_paths=max_paths_per_indication)
        all_positive_paths.extend(paths)
    typer.echo(f"Extracted {len(all_positive_paths)} positive paths")

    typer.echo("Generating negative examples...")
    negative_paths = generate_negatives(all_positive_paths)
    typer.echo(f"Generated {len(negative_paths)} negative paths")

    typer.echo("Splitting train/test by indication...")
    train_pos, test_pos = split_by_indication(all_positive_paths, test_size=test_size, random_seed=random_seed)
    train_neg, test_neg = split_by_indication(negative_paths, test_size=test_size, random_seed=random_seed)

    train_paths = train_pos + train_neg
    test_paths = test_pos + test_neg

    train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))
    test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))

    typer.echo(f"Train: {len(train_paths)} paths ({len(train_pos)} pos, {len(train_neg)} neg)")
    typer.echo(f"Test: {len(test_paths)} paths ({len(test_pos)} pos, {len(test_neg)} neg)")

    typer.echo("Loading API key...")
    api_key = load_api_key(api_key_path)

    typer.echo("Generating embeddings for training set...")
    train_embeddings = embed_paths(train_paths, api_key)
    typer.echo(f"Train embeddings shape: {train_embeddings.shape}")

    typer.echo("Generating embeddings for test set...")
    test_embeddings = embed_paths(test_paths, api_key)
    typer.echo(f"Test embeddings shape: {test_embeddings.shape}")

    typer.echo("Training Random Forest classifier...")
    model = train_classifier(train_embeddings, train_labels, random_state=random_seed)

    typer.echo("Evaluating on test set...")
    print_evaluation_report(model, test_embeddings, test_labels)

    typer.echo(f"Saving model to {output}...")
    with open(output, 'wb') as f:
        pickle.dump(model, f)

    typer.echo("Training complete!")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
