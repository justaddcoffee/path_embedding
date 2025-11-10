"""Tests for path extraction from multigraphs."""
import networkx as nx
from path_embedding.utils.path_extraction import build_multigraph, find_drug_disease_nodes, extract_paths
from path_embedding.datamodel.types import Path
from path_embedding.data.drugmechdb import load_drugmechdb


def test_build_multigraph():
    """Test building NetworkX multigraph from indication.

    >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    >>> graph = build_multigraph(indications[0])
    >>> isinstance(graph, nx.MultiDiGraph)
    True
    >>> graph.number_of_nodes()
    3
    >>> graph.number_of_edges()
    2
    """
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    indication = indications[0]

    graph = build_multigraph(indication)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_build_multigraph_node_attributes():
    """Test that nodes have correct attributes."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[0])

    # Check drug node
    drug_id = "MESH:D000068877"
    assert drug_id in graph.nodes
    assert graph.nodes[drug_id]["label"] == "Drug"
    assert graph.nodes[drug_id]["name"] == "imatinib"


def test_build_multigraph_edge_attributes():
    """Test that edges have correct attributes."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[0])

    # Check edge
    edges = list(graph.edges(data=True, keys=True))
    source, target, key, data = edges[0]

    assert data["key"] == "decreases activity of"


def test_find_drug_disease_nodes():
    """Test finding drug and disease nodes in graph.

    >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    >>> graph = build_multigraph(indications[0])
    >>> drug_id, disease_id = find_drug_disease_nodes(graph)
    >>> drug_id
    'MESH:D000068877'
    >>> disease_id
    'MESH:D015464'
    """
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[0])

    drug_id, disease_id = find_drug_disease_nodes(graph)

    assert drug_id == "MESH:D000068877"
    assert disease_id == "MESH:D015464"
    assert graph.nodes[drug_id]["label"] == "Drug"
    assert graph.nodes[disease_id]["label"] == "Disease"


def test_find_drug_disease_nodes_second_example():
    """Test with second example (multiple intermediate nodes)."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[1])

    drug_id, disease_id = find_drug_disease_nodes(graph)

    assert graph.nodes[drug_id]["label"] == "Drug"
    assert graph.nodes[disease_id]["label"] == "Disease"


def test_extract_paths_simple():
    """Test extracting paths from simple graph.

    >>> from path_embedding.data.drugmechdb import load_drugmechdb
    >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    >>> graph = build_multigraph(indications[0])
    >>> paths = extract_paths(graph, "DB00619_MESH_D015464_1")
    >>> len(paths)
    1
    >>> isinstance(paths[0], Path)
    True
    """
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    indication = indications[0]
    graph = build_multigraph(indication)

    paths = extract_paths(graph, indication["graph"]["_id"])

    assert len(paths) == 1
    assert isinstance(paths[0], Path)


def test_extract_paths_structure():
    """Test that extracted path has correct structure."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    indication = indications[0]
    graph = build_multigraph(indication)

    paths = extract_paths(graph, indication["graph"]["_id"])
    path = paths[0]

    # Should have 3 nodes: Drug -> Protein -> Disease
    assert len(path.nodes) == 3
    # Should have 2 edges
    assert len(path.edges) == 2

    # Check drug and disease
    assert path.drug_id == "MESH:D000068877"
    assert path.disease_id == "MESH:D015464"
    assert path.indication_id == "DB00619_MESH_D015464_1"

    # Check node order
    assert path.nodes[0].label == "Drug"
    assert path.nodes[1].label == "Protein"
    assert path.nodes[2].label == "Disease"


def test_extract_paths_multiple():
    """Test extracting multiple paths."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    indication = indications[1]  # Second example has longer path
    graph = build_multigraph(indication)

    paths = extract_paths(graph, indication["graph"]["_id"])

    # Should extract at least one path
    assert len(paths) >= 1

    # All paths should start with Drug and end with Disease
    for path in paths:
        assert path.nodes[0].label == "Drug"
        assert path.nodes[-1].label == "Disease"


def test_extract_paths_max_limit():
    """Test limiting number of paths extracted."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    indication = indications[0]
    graph = build_multigraph(indication)

    # Extract with limit
    paths = extract_paths(graph, indication["graph"]["_id"], max_paths=1)

    assert len(paths) <= 1
