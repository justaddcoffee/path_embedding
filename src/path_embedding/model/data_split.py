"""Train/test splitting by indication."""
from typing import List, Tuple
from collections import defaultdict
import random
from path_embedding.datamodel.types import Path


def split_by_indication(
    paths: List[Path],
    test_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """Split paths into train/test by indication (drug-disease pair).

    Groups paths by indication_id, then splits at indication level to
    prevent data leakage.

    Args:
        paths: List of Path objects
        test_size: Fraction for test set (default 0.2)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_paths, test_paths)

    Example:
        >>> from path_embedding.datamodel.types import Path
        >>> paths = [
        ...     Path(nodes=[], edges=[], drug_id="D1", disease_id="DIS1",
        ...          indication_id="ind1"),
        ...     Path(nodes=[], edges=[], drug_id="D2", disease_id="DIS2",
        ...          indication_id="ind2"),
        ... ]
        >>> train, test = split_by_indication(paths, test_size=0.5, random_seed=42)
        >>> len(train) + len(test) == 2
        True
    """
    random.seed(random_seed)

    # Group paths by indication_id
    indication_groups = defaultdict(list)
    for path in paths:
        indication_groups[path.indication_id].append(path)

    # Get list of indication IDs
    indication_ids = list(indication_groups.keys())

    # Shuffle and split indication IDs
    random.shuffle(indication_ids)
    split_point = int(len(indication_ids) * (1 - test_size))

    train_indication_ids = indication_ids[:split_point]
    test_indication_ids = indication_ids[split_point:]

    # Collect paths for train and test
    train_paths = []
    for ind_id in train_indication_ids:
        train_paths.extend(indication_groups[ind_id])

    test_paths = []
    for ind_id in test_indication_ids:
        test_paths.extend(indication_groups[ind_id])

    return train_paths, test_paths
