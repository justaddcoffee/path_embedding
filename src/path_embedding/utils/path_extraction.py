"""Path extraction from NetworkX multigraphs."""
from typing import Dict, Any, Tuple, List
import networkx as nx
import random
from path_embedding.datamodel.types import Node, Edge, Path


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


def extract_paths(
    graph: nx.MultiDiGraph,
    indication_id: str,
    max_paths: int = 10
) -> List[Path]:
    """Extract all simple paths from drug to disease in multigraph.

    Args:
        graph: NetworkX MultiDiGraph
        indication_id: Original DrugMechDB indication ID
        max_paths: Maximum number of paths to extract (randomly sample if more)

    Returns:
        List of Path objects

    Example:
        >>> from path_embedding.data.drugmechdb import load_drugmechdb
        >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
        >>> graph = build_multigraph(indications[0])
        >>> paths = extract_paths(graph, indications[0]["_id"])
        >>> len(paths) >= 1
        True
    """
    # Find drug and disease nodes
    drug_id, disease_id = find_drug_disease_nodes(graph)

    # Extract all simple paths
    all_node_paths = list(nx.all_simple_paths(graph, drug_id, disease_id))

    # Sample if too many paths
    if len(all_node_paths) > max_paths:
        all_node_paths = random.sample(all_node_paths, max_paths)

    # Convert to Path objects
    paths = []
    for node_path in all_node_paths:
        # Build nodes list
        nodes = []
        for node_id in node_path:
            node_data = graph.nodes[node_id]
            nodes.append(Node(
                id=node_id,
                label=node_data["label"],
                name=node_data["name"]
            ))

        # Build edges list
        edges = []
        for i in range(len(node_path) - 1):
            source = node_path[i]
            target = node_path[i + 1]

            # Get edge data (handle multigraph - may have multiple edges)
            edge_data = graph.get_edge_data(source, target)
            # Take first edge if multiple exist
            edge_key = list(edge_data.keys())[0]
            edge_attrs = edge_data[edge_key]

            edges.append(Edge(
                key=edge_attrs["key"],
                source=source,
                target=target
            ))

        paths.append(Path(
            nodes=nodes,
            edges=edges,
            drug_id=drug_id,
            disease_id=disease_id,
            indication_id=indication_id
        ))

    return paths
