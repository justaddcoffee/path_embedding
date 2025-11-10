"""Tests for core data types."""
from path_embedding.datamodel.types import Node, Edge, Path


def test_node_creation():
    """Test creating a Node.

    >>> node = Node(id="MESH:D001241", label="Drug", name="aspirin")
    >>> node.id
    'MESH:D001241'
    >>> node.label
    'Drug'
    >>> node.name
    'aspirin'
    """
    node = Node(id="MESH:D001241", label="Drug", name="aspirin")
    assert node.id == "MESH:D001241"
    assert node.label == "Drug"
    assert node.name == "aspirin"


def test_edge_creation():
    """Test creating an Edge.

    >>> edge = Edge(key="decreases activity of", source="MESH:D001241", target="UniProt:P00519")
    >>> edge.key
    'decreases activity of'
    """
    edge = Edge(
        key="decreases activity of",
        source="MESH:D001241",
        target="UniProt:P00519"
    )
    assert edge.key == "decreases activity of"
    assert edge.source == "MESH:D001241"
    assert edge.target == "UniProt:P00519"


def test_path_creation():
    """Test creating a Path."""
    nodes = [
        Node(id="MESH:D001241", label="Drug", name="aspirin"),
        Node(id="UniProt:P00519", label="Protein", name="COX2"),
        Node(id="MESH:D010146", label="Disease", name="pain"),
    ]
    edges = [
        Edge(key="inhibits", source="MESH:D001241", target="UniProt:P00519"),
        Edge(key="causes", source="UniProt:P00519", target="MESH:D010146"),
    ]
    path = Path(
        nodes=nodes,
        edges=edges,
        drug_id="MESH:D001241",
        disease_id="MESH:D010146",
        indication_id="test_indication_1"
    )
    assert len(path.nodes) == 3
    assert len(path.edges) == 2
    assert path.drug_id == "MESH:D001241"
    assert path.disease_id == "MESH:D010146"
