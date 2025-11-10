"""Generate negative examples via cross-disease shuffling."""
from typing import List, Dict
from collections import defaultdict
import random
from path_embedding.datamodel.types import Path, Node, Edge


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


def generate_negative_path(
    positive_path: Path,
    node_inventory: Dict[str, Dict[str, List[Node]]]
) -> Path:
    """Generate negative path via cross-disease shuffling with type matching.

    Keep drug and disease nodes. For each intermediate node, replace with
    a random node of the same type from a different disease context.

    Args:
        positive_path: Original positive Path
        node_inventory: Inventory from build_node_inventory()

    Returns:
        Negative Path with shuffled intermediate nodes

    Example:
        >>> random.seed(42)
        >>> from path_embedding.datamodel.types import Node, Edge, Path
        >>> nodes = [
        ...     Node(id="D1", label="Drug", name="drug1"),
        ...     Node(id="G1", label="Gene", name="gene1"),
        ...     Node(id="DIS1", label="Disease", name="disease1")
        ... ]
        >>> edges = [
        ...     Edge(key="reg", source="D1", target="G1"),
        ...     Edge(key="causes", source="G1", target="DIS1")
        ... ]
        >>> path = Path(nodes=nodes, edges=edges, drug_id="D1",
        ...             disease_id="DIS1", indication_id="ind1")
        >>> inv = {"Gene": {"DIS2": [Node(id="G2", label="Gene", name="gene2")]}}
        >>> neg = generate_negative_path(path, inv)
        >>> neg.nodes[0].id == "D1"  # Same drug
        True
    """
    new_nodes = []

    for i, node in enumerate(positive_path.nodes):
        # Keep drug and disease
        if node.label in ["Drug", "Disease"]:
            new_nodes.append(node)
        else:
            # Replace with random node of same type from different disease
            node_type = node.label

            if node_type in node_inventory:
                # Get all disease contexts for this node type
                disease_dict = node_inventory[node_type]

                # Exclude current disease context
                other_diseases = [
                    disease_id for disease_id in disease_dict.keys()
                    if disease_id != positive_path.disease_id
                ]

                if other_diseases:
                    # Pick random disease context
                    random_disease = random.choice(other_diseases)
                    # Pick random node from that context
                    random_node = random.choice(disease_dict[random_disease])
                    new_nodes.append(random_node)
                else:
                    # No other disease context, keep original
                    new_nodes.append(node)
            else:
                # Node type not in inventory, keep original
                new_nodes.append(node)

    # Build new edges with updated source/target IDs
    new_edges = []
    for i, edge in enumerate(positive_path.edges):
        new_edges.append(Edge(
            key=edge.key,
            source=new_nodes[i].id,
            target=new_nodes[i + 1].id
        ))

    # Create negative path
    negative_path = Path(
        nodes=new_nodes,
        edges=new_edges,
        drug_id=positive_path.drug_id,
        disease_id=positive_path.disease_id,
        indication_id=f"{positive_path.indication_id}_negative"
    )

    return negative_path


def generate_negatives(positive_paths: List[Path]) -> List[Path]:
    """Generate negative examples for all positive paths.

    Creates 1:1 ratio of positives to negatives using cross-disease shuffling.

    Args:
        positive_paths: List of positive Path objects

    Returns:
        List of negative Path objects (same length as input)

    Example:
        >>> random.seed(42)
        >>> from path_embedding.datamodel.types import Node, Edge, Path
        >>> nodes = [
        ...     Node(id="D1", label="Drug", name="d1"),
        ...     Node(id="G1", label="Gene", name="g1"),
        ...     Node(id="DIS1", label="Disease", name="dis1")
        ... ]
        >>> edges = [Edge(key="r", source="D1", target="G1"),
        ...          Edge(key="c", source="G1", target="DIS1")]
        >>> path = Path(nodes=nodes, edges=edges, drug_id="D1",
        ...             disease_id="DIS1", indication_id="i1")
        >>> negs = generate_negatives([path])
        >>> len(negs)
        1
    """
    # Build node inventory from all positive paths
    inventory = build_node_inventory(positive_paths)

    # Generate one negative for each positive
    negatives = []
    for positive_path in positive_paths:
        negative_path = generate_negative_path(positive_path, inventory)
        negatives.append(negative_path)

    return negatives
