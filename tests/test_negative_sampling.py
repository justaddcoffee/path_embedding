"""Tests for negative example generation."""
from path_embedding.data.negative_sampling import build_node_inventory
from path_embedding.data.drugmechdb import load_drugmechdb
from path_embedding.utils.path_extraction import build_multigraph, extract_paths


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
