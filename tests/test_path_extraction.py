"""Tests for path extraction from multigraphs."""
import networkx as nx
from path_embedding.utils.path_extraction import build_multigraph, find_drug_disease_nodes
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
