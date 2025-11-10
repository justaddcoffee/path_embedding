"""Tests for negative example generation."""
from path_embedding.data.negative_sampling import (
    build_node_inventory,
    generate_negative_path,
    generate_negatives,
)
from path_embedding.data.drugmechdb import load_drugmechdb
from path_embedding.utils.path_extraction import build_multigraph, extract_paths
import random


def test_build_node_inventory():
    """Test building node inventory grouped by type and disease.

    >>> from path_embedding.datamodel.types import Node, Path
    >>> nodes1 = [
    ...     Node(id="D1", label="Drug", name="drug1"),
    ...     Node(id="G1", label="Gene", name="gene1"),
    ...     Node(id="DIS1", label="Disease", name="disease1")
    ... ]
    >>> path1 = Path(nodes=nodes1, edges=[], drug_id="D1",
    ...              disease_id="DIS1", indication_id="ind1")
    >>> inventory = build_node_inventory([path1])
    >>> "Gene" in inventory
    True
    """
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")

    # Extract all paths
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)

    # Should have node types as keys
    assert isinstance(inventory, dict)
    assert "Protein" in inventory or "BiologicalProcess" in inventory

    # Each node type should map to disease -> nodes
    for node_type, disease_dict in inventory.items():
        assert isinstance(disease_dict, dict)


def test_build_node_inventory_structure():
    """Test inventory structure: {node_type: {disease_id: [nodes]}}."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)

    # Check structure
    for node_type, disease_dict in inventory.items():
        for disease_id, nodes in disease_dict.items():
            assert isinstance(nodes, list)
            assert len(nodes) > 0
            # All nodes should have the expected type
            for node in nodes:
                assert node.label == node_type


def test_build_node_inventory_excludes_drug_disease():
    """Test that Drug and Disease nodes are excluded from inventory."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)

    # Should not include Drug or Disease in inventory
    assert "Drug" not in inventory
    assert "Disease" not in inventory


def test_generate_negative_path():
    """Test generating negative path from positive.

    >>> random.seed(42)
    >>> from path_embedding.datamodel.types import Node, Edge, Path
    >>> nodes = [
    ...     Node(id="D1", label="Drug", name="drug1"),
    ...     Node(id="G1", label="Gene", name="gene1"),
    ...     Node(id="DIS1", label="Disease", name="disease1")
    ... ]
    >>> edges = [
    ...     Edge(key="regulates", source="D1", target="G1"),
    ...     Edge(key="causes", source="G1", target="DIS1")
    ... ]
    >>> path = Path(nodes=nodes, edges=edges, drug_id="D1",
    ...             disease_id="DIS1", indication_id="ind1")
    >>> inventory = {"Gene": {"DIS2": [Node(id="G2", label="Gene", name="gene2")]}}
    >>> neg_path = generate_negative_path(path, inventory)
    >>> neg_path.nodes[0].id == "D1"  # Same drug
    True
    >>> neg_path.nodes[-1].id == "DIS1"  # Same disease
    True
    """
    random.seed(42)

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)
    positive_path = all_paths[0]

    negative_path = generate_negative_path(positive_path, inventory)

    # Should preserve drug and disease
    assert negative_path.drug_id == positive_path.drug_id
    assert negative_path.disease_id == positive_path.disease_id

    # Should have same number of nodes
    assert len(negative_path.nodes) == len(positive_path.nodes)

    # First and last nodes should be same
    assert negative_path.nodes[0].id == positive_path.nodes[0].id
    assert negative_path.nodes[-1].id == positive_path.nodes[-1].id


def test_generate_negative_path_preserves_types():
    """Test that negative path preserves node types."""
    random.seed(42)

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)

    # Use second example which has more intermediate nodes
    positive_path = all_paths[1] if len(all_paths) > 1 else all_paths[0]
    negative_path = generate_negative_path(positive_path, inventory)

    # Check that node types match in order
    for i, (pos_node, neg_node) in enumerate(zip(positive_path.nodes, negative_path.nodes)):
        assert pos_node.label == neg_node.label, f"Node {i} type mismatch"


def test_generate_negative_path_different_disease_context():
    """Test that intermediate nodes come from different disease context."""
    random.seed(42)

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")

    # Need at least 2 different diseases
    if len(indications) < 2:
        return  # Skip if not enough data

    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)
    positive_path = all_paths[0]

    negative_path = generate_negative_path(positive_path, inventory)

    # At least one intermediate node should be different
    # (if we have enough diversity in the data)
    intermediate_changed = False
    for i in range(1, len(positive_path.nodes) - 1):
        if positive_path.nodes[i].id != negative_path.nodes[i].id:
            intermediate_changed = True
            break

    # If we have multiple diseases, should have changed something
    if len(indications) >= 2:
        assert intermediate_changed


def test_generate_negatives():
    """Test generating negative dataset.

    >>> random.seed(42)
    >>> from path_embedding.datamodel.types import Node, Edge, Path
    >>> nodes = [
    ...     Node(id="D1", label="Drug", name="drug1"),
    ...     Node(id="G1", label="Gene", name="gene1"),
    ...     Node(id="DIS1", label="Disease", name="disease1")
    ... ]
    >>> edges = [Edge(key="reg", source="D1", target="G1"),
    ...          Edge(key="causes", source="G1", target="DIS1")]
    >>> path = Path(nodes=nodes, edges=edges, drug_id="D1",
    ...             disease_id="DIS1", indication_id="ind1")
    >>> negs = generate_negatives([path])
    >>> len(negs) == 1
    True
    """
    random.seed(42)

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    negatives = generate_negatives(all_paths)

    # Should have 1:1 ratio
    assert len(negatives) == len(all_paths)

    # Each negative should be valid Path
    for neg in negatives:
        from path_embedding.datamodel.types import Path
        assert isinstance(neg, Path)
        assert len(neg.nodes) > 0
        assert neg.nodes[0].label == "Drug"
        assert neg.nodes[-1].label == "Disease"
