"""Core data types for path representation."""
from dataclasses import dataclass
from typing import List


@dataclass
class Node:
    """A node in a mechanistic path.

    Attributes:
        id: Unique identifier (e.g., MESH:D001241)
        label: Node type (e.g., Drug, Protein, Disease)
        name: Human-readable name (e.g., aspirin)
    """
    id: str
    label: str
    name: str


@dataclass
class Edge:
    """An edge in a mechanistic path.

    Attributes:
        key: Relationship type (e.g., "decreases activity of")
        source: Source node ID
        target: Target node ID
    """
    key: str
    source: str
    target: str


@dataclass
class Path:
    """A complete mechanistic path from drug to disease.

    Attributes:
        nodes: Ordered list of nodes in the path
        edges: Ordered list of edges in the path
        drug_id: ID of the drug (start) node
        disease_id: ID of the disease (end) node
        indication_id: Original DrugMechDB indication ID
    """
    nodes: List[Node]
    edges: List[Edge]
    drug_id: str
    disease_id: str
    indication_id: str
