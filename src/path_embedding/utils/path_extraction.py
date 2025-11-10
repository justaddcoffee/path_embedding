"""Path extraction from NetworkX multigraphs."""
from typing import Dict, Any, Tuple
import networkx as nx


def build_multigraph(indication: Dict[str, Any]) -> nx.MultiDiGraph:
    """Build NetworkX MultiDiGraph from DrugMechDB indication.

    Args:
        indication: DrugMechDB indication entry with nodes and links

    Returns:
        NetworkX MultiDiGraph with nodes and edges

    Example:
        >>> from path_embedding.data.drugmechdb import load_drugmechdb
        >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
        >>> graph = build_multigraph(indications[0])
        >>> graph.number_of_nodes() >= 3
        True
    """
    graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # Add nodes with attributes
    for node in indication["nodes"]:
        graph.add_node(
            node["id"],
            label=node["label"],
            name=node["name"]
        )

    # Add edges with attributes
    for link in indication["links"]:
        # For MultiDiGraph, we use the edge relationship type as the key
        # and also store it as a data attribute for convenient access
        graph.add_edge(
            link["source"],
            link["target"],
            key=link["key"]
        )
        # Also store the key in edge attributes for easier access
        edge_data = graph.get_edge_data(link["source"], link["target"])
        if link["key"] in edge_data:
            edge_data[link["key"]]["key"] = link["key"]

    return graph


def find_drug_disease_nodes(graph: nx.MultiDiGraph) -> Tuple[str, str]:
    """Find drug and disease node IDs in graph.

    Args:
        graph: NetworkX MultiDiGraph with labeled nodes

    Returns:
        Tuple of (drug_id, disease_id)

    Raises:
        ValueError: If drug or disease node not found

    Example:
        >>> from path_embedding.data.drugmechdb import load_drugmechdb
        >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
        >>> graph = build_multigraph(indications[0])
        >>> drug_id, disease_id = find_drug_disease_nodes(graph)
        >>> isinstance(drug_id, str) and isinstance(disease_id, str)
        True
    """
    drug_id: str | None = None
    disease_id: str | None = None

    for node_id, attrs in graph.nodes(data=True):
        if attrs["label"] == "Drug":
            drug_id = node_id
        elif attrs["label"] == "Disease":
            disease_id = node_id

    if drug_id is None:
        raise ValueError("No Drug node found in graph")
    if disease_id is None:
        raise ValueError("No Disease node found in graph")

    return drug_id, disease_id
