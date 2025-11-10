"""Tests for path to text conversion."""
from path_embedding.embedding.text_formatter import path_to_text
from path_embedding.datamodel.types import Node, Edge, Path


def test_path_to_text_simple():
    """Test converting simple path to text.

    >>> nodes = [
    ...     Node(id="MESH:D001241", label="Drug", name="aspirin"),
    ...     Node(id="UniProt:P00519", label="Protein", name="COX2"),
    ...     Node(id="MESH:D010146", label="Disease", name="pain")
    ... ]
    >>> edges = [
    ...     Edge(key="inhibits", source="MESH:D001241", target="UniProt:P00519"),
    ...     Edge(key="causes", source="UniProt:P00519", target="MESH:D010146")
    ... ]
    >>> path = Path(nodes=nodes, edges=edges, drug_id="MESH:D001241",
    ...             disease_id="MESH:D010146", indication_id="test")
    >>> text = path_to_text(path)
    >>> text
    'Drug: aspirin | inhibits | Protein: COX2 | causes | Disease: pain'
    """
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
        indication_id="test_1"
    )

    text = path_to_text(path)

    expected = "Drug: aspirin | inhibits | Protein: COX2 | causes | Disease: pain"
    assert text == expected


def test_path_to_text_real_example():
    """Test with real DrugMechDB example."""
    from path_embedding.data.drugmechdb import load_drugmechdb
    from path_embedding.utils.path_extraction import build_multigraph, extract_paths

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[0])
    paths = extract_paths(graph, indications[0]["graph"]["_id"])

    text = path_to_text(paths[0])

    # Should contain all components
    assert "Drug:" in text
    assert "Protein:" in text
    assert "Disease:" in text
    assert "|" in text
    assert "decreases activity of" in text or "causes" in text


def test_path_to_text_format():
    """Test text format structure."""
    nodes = [
        Node(id="A", label="Drug", name="drugA"),
        Node(id="B", label="Gene", name="geneB"),
        Node(id="C", label="Disease", name="diseaseC"),
    ]
    edges = [
        Edge(key="regulates", source="A", target="B"),
        Edge(key="affects", source="B", target="C"),
    ]
    path = Path(
        nodes=nodes, edges=edges,
        drug_id="A", disease_id="C", indication_id="test"
    )

    text = path_to_text(path)

    # Check format: Node | Edge | Node | Edge | Node
    parts = [p.strip() for p in text.split("|")]
    assert len(parts) == 5
    assert parts[0].startswith("Drug:")
    assert parts[2].startswith("Gene:")
    assert parts[4].startswith("Disease:")
