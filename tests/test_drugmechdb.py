"""Tests for DrugMechDB data loading."""
from path_embedding.data.drugmechdb import load_drugmechdb


def test_load_drugmechdb():
    """Test loading DrugMechDB YAML file.

    >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    >>> len(indications)
    2
    >>> indications[0]["_id"]
    'DB00619_MESH_D015464_1'
    """
    test_file = "tests/data/sample_drugmechdb.yaml"
    indications = load_drugmechdb(test_file)

    assert len(indications) == 2
    assert indications[0]["_id"] == "DB00619_MESH_D015464_1"
    assert indications[1]["_id"] == "DB00316_MESH_D010146_1"


def test_load_drugmechdb_structure():
    """Test that loaded data has correct structure."""
    test_file = "tests/data/sample_drugmechdb.yaml"
    indications = load_drugmechdb(test_file)

    indication = indications[0]
    assert "nodes" in indication
    assert "links" in indication
    assert "graph" in indication
    assert indication["directed"] is True
    assert indication["multigraph"] is True


def test_load_drugmechdb_nodes():
    """Test node structure."""
    test_file = "tests/data/sample_drugmechdb.yaml"
    indications = load_drugmechdb(test_file)

    nodes = indications[0]["nodes"]
    assert len(nodes) == 3

    drug_node = nodes[0]
    assert drug_node["id"] == "MESH:D000068877"
    assert drug_node["label"] == "Drug"
    assert drug_node["name"] == "imatinib"


def test_load_drugmechdb_edges():
    """Test edge structure."""
    test_file = "tests/data/sample_drugmechdb.yaml"
    indications = load_drugmechdb(test_file)

    links = indications[0]["links"]
    assert len(links) == 2

    edge = links[0]
    assert edge["key"] == "decreases activity of"
    assert edge["source"] == "MESH:D000068877"
    assert edge["target"] == "UniProt:P00519"
