"""Tests for KGX data loading (stub)."""
import pytest
from path_embedding.data.kgx import load_kgx_paths


def test_load_kgx_paths_not_implemented():
    """Test that KGX loader raises NotImplementedError.

    >>> load_kgx_paths("dummy.json")
    Traceback (most recent call last):
        ...
    NotImplementedError: KGX support not yet implemented
    """
    with pytest.raises(NotImplementedError):
        load_kgx_paths("dummy.json")
