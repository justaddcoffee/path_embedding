"""Tests for train/test splitting."""
from path_embedding.model.data_split import split_by_indication
from path_embedding.data.drugmechdb import load_drugmechdb
from path_embedding.utils.path_extraction import build_multigraph, extract_paths


def test_split_by_indication():
    """Test splitting paths by indication.

    >>> from path_embedding.datamodel.types import Node, Edge, Path
    >>> paths = [
    ...     Path(nodes=[], edges=[], drug_id="D1", disease_id="DIS1",
    ...          indication_id="ind1"),
    ...     Path(nodes=[], edges=[], drug_id="D1", disease_id="DIS1",
    ...          indication_id="ind1"),
    ...     Path(nodes=[], edges=[], drug_id="D2", disease_id="DIS2",
    ...          indication_id="ind2"),
    ... ]
    >>> train, test = split_by_indication(paths, test_size=0.5, random_seed=42)
    >>> len(train) + len(test) == 3
    True
    """
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")

    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["graph"]["_id"])
        all_paths.extend(paths)

    train_paths, test_paths = split_by_indication(
        all_paths,
        test_size=0.2,
        random_seed=42
    )

    # Should have train and test
    assert len(train_paths) > 0
    assert len(test_paths) > 0

    # Total should equal input
    assert len(train_paths) + len(test_paths) == len(all_paths)


def test_split_by_indication_no_leakage():
    """Test that same indication doesn't appear in both train and test."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")

    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["graph"]["_id"])
        all_paths.extend(paths)

    train_paths, test_paths = split_by_indication(
        all_paths,
        test_size=0.5,
        random_seed=42
    )

    # Get indication IDs from train and test
    train_indications = set(p.indication_id for p in train_paths)
    test_indications = set(p.indication_id for p in test_paths)

    # Should have no overlap
    assert len(train_indications & test_indications) == 0


def test_split_by_indication_reproducible():
    """Test that split is reproducible with same seed."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")

    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["graph"]["_id"])
        all_paths.extend(paths)

    train1, test1 = split_by_indication(all_paths, test_size=0.5, random_seed=42)
    train2, test2 = split_by_indication(all_paths, test_size=0.5, random_seed=42)

    # Same seed should give same split
    assert len(train1) == len(train2)
    assert len(test1) == len(test2)

    train_ids_1 = [p.indication_id for p in train1]
    train_ids_2 = [p.indication_id for p in train2]
    assert train_ids_1 == train_ids_2
