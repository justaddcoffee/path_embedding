"""Convert paths to structured text representations."""
from path_embedding.datamodel.types import Path


def path_to_text(path: Path) -> str:
    """Convert a Path object to structured text format.

    Format: {label}: {name} | {edge_key} | {label}: {name} | ...

    Args:
        path: Path object to convert

    Returns:
        Structured text representation

    Example:
        >>> from path_embedding.datamodel.types import Node, Edge, Path
        >>> nodes = [
        ...     Node(id="A", label="Drug", name="aspirin"),
        ...     Node(id="B", label="Protein", name="COX2"),
        ...     Node(id="C", label="Disease", name="pain")
        ... ]
        >>> edges = [
        ...     Edge(key="inhibits", source="A", target="B"),
        ...     Edge(key="causes", source="B", target="C")
        ... ]
        >>> path = Path(nodes=nodes, edges=edges, drug_id="A",
        ...             disease_id="C", indication_id="test")
        >>> path_to_text(path)
        'Drug: aspirin | inhibits | Protein: COX2 | causes | Disease: pain'
    """
    parts = []

    for i, node in enumerate(path.nodes):
        # Add node
        parts.append(f"{node.label}: {node.name}")

        # Add edge if not last node
        if i < len(path.edges):
            parts.append(path.edges[i].key)

    return " | ".join(parts)
