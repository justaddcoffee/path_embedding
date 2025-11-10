"""Tests for OpenAI embedding generation."""
import numpy as np
from path_embedding.embedding.openai_embedder import load_api_key, embed_text


def test_load_api_key():
    """Test loading API key from file.

    >>> key = load_api_key("/Users/jtr4v/openai.key.another")
    >>> len(key) > 0
    True
    >>> key.startswith("sk-")
    True
    """
    key_path = "/Users/jtr4v/openai.key.another"
    key = load_api_key(key_path)

    assert isinstance(key, str)
    assert len(key) > 0
    # OpenAI keys start with "sk-"
    assert key.startswith("sk-")


def test_embed_text():
    """Test embedding simple text with OpenAI API.

    Note: This test makes real API call and may be slow.
    """
    key = load_api_key("/Users/jtr4v/openai.key.another")
    text = "Drug: aspirin | inhibits | Protein: COX2"

    embedding = embed_text(text, key)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1  # 1D array
    assert embedding.shape[0] > 0  # Has dimensions
    # text-embedding-3-small has 1536 dimensions
    assert embedding.shape[0] == 1536


def test_embed_text_different_inputs():
    """Test that different texts produce different embeddings."""
    key = load_api_key("/Users/jtr4v/openai.key.another")

    text1 = "Drug: aspirin | inhibits | Protein: COX2"
    text2 = "Drug: imatinib | decreases activity of | Protein: BCR/ABL"

    emb1 = embed_text(text1, key)
    emb2 = embed_text(text2, key)

    # Embeddings should be different
    assert not np.array_equal(emb1, emb2)


def test_embed_paths():
    """Test embedding multiple paths.

    Note: Makes real API calls, may be slow.
    """
    from path_embedding.embedding.openai_embedder import embed_paths
    from path_embedding.data.drugmechdb import load_drugmechdb
    from path_embedding.utils.path_extraction import build_multigraph, extract_paths

    key = load_api_key("/Users/jtr4v/openai.key.another")

    # Load sample paths
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        indication_paths = extract_paths(graph, indication["_id"], max_paths=1)
        paths.extend(indication_paths)

    embeddings = embed_paths(paths, key)

    # Should return 2D array: n_paths x embedding_dim
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings.shape) == 2
    assert embeddings.shape[0] == len(paths)
    assert embeddings.shape[1] == 1536  # text-embedding-3-small dimension


def test_embed_paths_integration():
    """Integration test with text formatter."""
    from path_embedding.embedding.openai_embedder import embed_paths
    from path_embedding.data.drugmechdb import load_drugmechdb
    from path_embedding.utils.path_extraction import build_multigraph, extract_paths

    key = load_api_key("/Users/jtr4v/openai.key.another")

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[0])
    paths = extract_paths(graph, indications[0]["_id"])

    embeddings = embed_paths(paths, key)

    assert embeddings.shape[0] == len(paths)
