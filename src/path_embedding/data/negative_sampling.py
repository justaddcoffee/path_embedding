"""Generate negative examples via cross-disease shuffling."""
from typing import List, Dict
from collections import defaultdict
from path_embedding.datamodel.types import Path, Node


def build_node_inventory(paths: List[Path]) -> Dict[str, Dict[str, List[Node]]]:
    """Build inventory of nodes grouped by type and disease context.

    Structure: {node_type: {disease_id: [nodes]}}
    Excludes Drug and Disease nodes (only intermediate nodes).

    Args:
        paths: List of Path objects

    Returns:
        Nested dict mapping node_type -> disease_id -> list of nodes

    Example:
        >>> from path_embedding.datamodel.types import Node, Path
        >>> nodes = [
        ...     Node(id="D1", label="Drug", name="drug1"),
        ...     Node(id="G1", label="Gene", name="gene1"),
        ...     Node(id="DIS1", label="Disease", name="disease1")
        ... ]
        >>> path = Path(nodes=nodes, edges=[], drug_id="D1",
        ...             disease_id="DIS1", indication_id="ind1")
        >>> inventory = build_node_inventory([path])
        >>> "Gene" in inventory
        True
        >>> "Drug" not in inventory
        True
    """
    inventory: Dict[str, Dict[str, List[Node]]] = defaultdict(lambda: defaultdict(list))

    for path in paths:
        # Get disease context from this path
        disease_id = path.disease_id

        # Add all intermediate nodes (exclude Drug and Disease)
        for node in path.nodes:
            if node.label not in ["Drug", "Disease"]:
                # Add to inventory: node_type -> disease_id -> nodes
                inventory[node.label][disease_id].append(node)

    # Convert to regular dict
    return {k: dict(v) for k, v in inventory.items()}
