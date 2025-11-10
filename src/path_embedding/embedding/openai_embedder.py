"""OpenAI embedding generation."""
from typing import List
import numpy as np
from openai import OpenAI

from path_embedding.datamodel.types import Path
from path_embedding.embedding.text_formatter import path_to_text


def load_api_key(key_path: str) -> str:
    """Load OpenAI API key from file.

    Args:
        key_path: Path to file containing API key

    Returns:
        API key as string

    Example:
        >>> key = load_api_key("/Users/jtr4v/openai.key.another")
        >>> len(key) > 0
        True
    """
    with open(key_path, 'r') as f:
        api_key = f.read().strip()
    return api_key


def embed_text(text: str, api_key: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Generate embedding for text using OpenAI API.

    Args:
        text: Text to embed
        api_key: OpenAI API key
        model: OpenAI embedding model (default: text-embedding-3-small)

    Returns:
        Embedding vector as numpy array

    Example:
        >>> key = load_api_key("/Users/jtr4v/openai.key.another")
        >>> emb = embed_text("test text", key)
        >>> isinstance(emb, np.ndarray)
        True
    """
    client = OpenAI(api_key=api_key)

    response = client.embeddings.create(
        input=text,
        model=model
    )

    embedding = np.array(response.data[0].embedding)
    return embedding


def embed_paths(
    paths: List[Path],
    api_key: str,
    model: str = "text-embedding-3-small"
) -> np.ndarray:
    """Generate embeddings for multiple paths.

    Args:
        paths: List of Path objects
        api_key: OpenAI API key
        model: OpenAI embedding model

    Returns:
        2D numpy array of shape (n_paths, embedding_dim)

    Example:
        >>> from path_embedding.datamodel.types import Node, Edge, Path
        >>> nodes = [
        ...     Node(id="A", label="Drug", name="aspirin"),
        ...     Node(id="B", label="Protein", name="COX2"),
        ...     Node(id="C", label="Disease", name="pain")
        ... ]
        >>> edges = [
        ...     Edge(key="inhibits", source="A", target="B"),
        ...     Edge(key="causes", source="B", target="C")
        ... ]
        >>> path = Path(nodes=nodes, edges=edges, drug_id="A",
        ...             disease_id="C", indication_id="test")
        >>> key = load_api_key("/Users/jtr4v/openai.key.another")
        >>> embs = embed_paths([path], key)
        >>> embs.shape[0] == 1
        True
    """
    embeddings = []

    for path in paths:
        # Convert path to text
        text = path_to_text(path)

        # Generate embedding
        embedding = embed_text(text, api_key, model)
        embeddings.append(embedding)

    # Stack into 2D array
    return np.vstack(embeddings)
