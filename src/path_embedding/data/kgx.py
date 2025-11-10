"""KGX format data loading (stub for future implementation).

KGX (Knowledge Graph Exchange) format support will be added in a future version.
This module provides stubs to document the planned interface.

Expected KGX format:
- Nodes: JSON/TSV with id, category, name fields
- Edges: JSON/TSV with subject, predicate, object fields

Future implementation will:
1. Load KGX nodes and edges
2. Construct paths from edge sequences
3. Convert to Path objects matching DrugMechDB format
"""
from typing import List
from path_embedding.datamodel.types import Path


def load_kgx_paths(file_path: str) -> List[Path]:
    """Load paths from KGX format file.

    Args:
        file_path: Path to KGX JSON or TSV file

    Returns:
        List of Path objects

    Raises:
        NotImplementedError: KGX support not yet implemented

    Example:
        >>> load_kgx_paths("paths.kgx.json")
        Traceback (most recent call last):
            ...
        NotImplementedError: KGX support not yet implemented
    """
    raise NotImplementedError("KGX support not yet implemented")
