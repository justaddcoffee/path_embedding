"""DrugMechDB data loading utilities."""
from typing import List, Dict, Any
import yaml


def load_drugmechdb(file_path: str) -> List[Dict[str, Any]]:
    """Load DrugMechDB YAML file.

    Args:
        file_path: Path to DrugMechDB YAML file

    Returns:
        List of indication entries (multigraphs)

    Example:
        >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
        >>> len(indications) >= 1
        True
        >>> "nodes" in indications[0]
        True
    """
    with open(file_path, 'r') as f:
        indications = yaml.safe_load(f)

    return indications
