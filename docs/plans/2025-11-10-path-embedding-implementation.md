# Path Embedding Classifier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a binary classifier that distinguishes biologically plausible from implausible drug-disease mechanistic paths using LLM embeddings.

**Architecture:** Extract paths from DrugMechDB multigraphs, convert to structured text, embed with OpenAI API, train Random Forest classifier. Generate negative examples via cross-disease shuffling with type matching. Split train/test at indication level to prevent leakage.

**Tech Stack:** NetworkX (multigraphs), OpenAI embeddings API, scikit-learn (Random Forest), pytest (TDD), typer (CLI)

**Design Document:** [2025-11-10-path-embedding-classifier-design.md](2025-11-10-path-embedding-classifier-design.md)

---

## Prerequisites

### Task 0: Add Dependencies

**Step 1: Add required dependencies**

Run:
```bash
uv add networkx openai numpy scikit-learn pyyaml requests
```

Expected: Dependencies added to pyproject.toml and installed

**Step 2: Verify installation**

Run:
```bash
uv run python -c "import networkx, openai, numpy, sklearn, yaml, requests; print('All deps OK')"
```

Expected: "All deps OK" printed

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add networkx, openai, sklearn, pyyaml, requests"
```

---

## Phase 1: Data Models and Path Extraction

### Task 1.1: Define Core Data Models

**Files:**
- Create: `src/path_embedding/datamodel/types.py`
- Test: `tests/test_types.py`

**Step 1: Write the failing test**

Create file `tests/test_types.py`:

```python
"""Tests for core data types."""
from path_embedding.datamodel.types import Node, Edge, Path


def test_node_creation():
    """Test creating a Node.

    >>> node = Node(id="MESH:D001241", label="Drug", name="aspirin")
    >>> node.id
    'MESH:D001241'
    >>> node.label
    'Drug'
    >>> node.name
    'aspirin'
    """
    node = Node(id="MESH:D001241", label="Drug", name="aspirin")
    assert node.id == "MESH:D001241"
    assert node.label == "Drug"
    assert node.name == "aspirin"


def test_edge_creation():
    """Test creating an Edge.

    >>> edge = Edge(key="decreases activity of", source="MESH:D001241", target="UniProt:P00519")
    >>> edge.key
    'decreases activity of'
    """
    edge = Edge(
        key="decreases activity of",
        source="MESH:D001241",
        target="UniProt:P00519"
    )
    assert edge.key == "decreases activity of"
    assert edge.source == "MESH:D001241"
    assert edge.target == "UniProt:P00519"


def test_path_creation():
    """Test creating a Path."""
    nodes = [
        Node(id="MESH:D001241", label="Drug", name="aspirin"),
        Node(id="UniProt:P00519", label="Protein", name="COX2"),
        Node(id="MESH:D010146", label="Disease", name="pain"),
    ]
    edges = [
        Edge(key="inhibits", source="MESH:D001241", target="UniProt:P00519"),
        Edge(key="causes", source="UniProt:P00519", target="MESH:D010146"),
    ]
    path = Path(
        nodes=nodes,
        edges=edges,
        drug_id="MESH:D001241",
        disease_id="MESH:D010146",
        indication_id="test_indication_1"
    )
    assert len(path.nodes) == 3
    assert len(path.edges) == 2
    assert path.drug_id == "MESH:D001241"
    assert path.disease_id == "MESH:D010146"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'path_embedding.datamodel'"

**Step 3: Write minimal implementation**

Create directory: `mkdir -p src/path_embedding/datamodel`

Create file `src/path_embedding/datamodel/__init__.py`:
```python
"""Data models for path embedding."""
```

Create file `src/path_embedding/datamodel/types.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`

Expected: PASS (all tests)

**Step 5: Run doctests**

Run: `uv run pytest tests/test_types.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/datamodel/ tests/test_types.py
git commit -m "feat: add core data models (Node, Edge, Path)"
```

---

### Task 1.2: DrugMechDB Data Loader

**Files:**
- Create: `src/path_embedding/data/__init__.py`
- Create: `src/path_embedding/data/drugmechdb.py`
- Test: `tests/test_drugmechdb.py`
- Create: `tests/data/sample_drugmechdb.yaml` (fixture)

**Step 1: Create test fixture**

Create file `tests/data/sample_drugmechdb.yaml`:

```yaml
- _id: DB00619_MESH_D015464_1
  directed: true
  graph:
    disease: Leukemia, Myelogenous, Chronic, BCR-ABL Positive
    disease_mesh: MESH:D015464
    drug: imatinib
    drug_mesh: MESH:D000068877
    drugbank: DB00619
  links:
  - key: decreases activity of
    source: MESH:D000068877
    target: UniProt:P00519
  - key: causes
    source: UniProt:P00519
    target: MESH:D015464
  multigraph: true
  nodes:
  - id: MESH:D000068877
    label: Drug
    name: imatinib
  - id: UniProt:P00519
    label: Protein
    name: BCR/ABL
  - id: MESH:D015464
    label: Disease
    name: CML (ph+)
- _id: DB00316_MESH_D010146_1
  directed: true
  graph:
    disease: Pain
    disease_mesh: MESH:D010146
    drug: Acetaminophen
    drug_mesh: MESH:D000082
    drugbank: DB00316
  links:
  - key: decreases activity of
    source: MESH:D000082
    target: UniProt:P23219
  - key: positively regulates
    source: UniProt:P23219
    target: GO:0001516
  - key: positively regulates
    source: GO:0001516
    target: MESH:D015232
  - key: positively correlated with
    source: MESH:D015232
    target: MESH:D010146
  multigraph: true
  nodes:
  - id: MESH:D000082
    label: Drug
    name: Acetaminophen
  - id: UniProt:P23219
    label: Protein
    name: PTGS2
  - id: GO:0001516
    label: BiologicalProcess
    name: prostaglandin biosynthetic process
  - id: MESH:D015232
    label: ChemicalSubstance
    name: Dinoprostone
  - id: MESH:D010146
    label: Disease
    name: Pain
```

**Step 2: Write the failing test**

Create file `tests/test_drugmechdb.py`:

```python
"""Tests for DrugMechDB data loading."""
from pathlib import Path
from path_embedding.data.drugmechdb import load_drugmechdb


def test_load_drugmechdb():
    """Test loading DrugMechDB YAML file.

    >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    >>> len(indications)
    2
    >>> indications[0]["_id"]
    'DB00619_MESH_D015464_1'
    """
    test_file = "tests/data/sample_drugmechdb.yaml"
    indications = load_drugmechdb(test_file)

    assert len(indications) == 2
    assert indications[0]["_id"] == "DB00619_MESH_D015464_1"
    assert indications[1]["_id"] == "DB00316_MESH_D010146_1"


def test_load_drugmechdb_structure():
    """Test that loaded data has correct structure."""
    test_file = "tests/data/sample_drugmechdb.yaml"
    indications = load_drugmechdb(test_file)

    indication = indications[0]
    assert "nodes" in indication
    assert "links" in indication
    assert "graph" in indication
    assert indication["directed"] is True
    assert indication["multigraph"] is True


def test_load_drugmechdb_nodes():
    """Test node structure."""
    test_file = "tests/data/sample_drugmechdb.yaml"
    indications = load_drugmechdb(test_file)

    nodes = indications[0]["nodes"]
    assert len(nodes) == 3

    drug_node = nodes[0]
    assert drug_node["id"] == "MESH:D000068877"
    assert drug_node["label"] == "Drug"
    assert drug_node["name"] == "imatinib"


def test_load_drugmechdb_edges():
    """Test edge structure."""
    test_file = "tests/data/sample_drugmechdb.yaml"
    indications = load_drugmechdb(test_file)

    links = indications[0]["links"]
    assert len(links) == 2

    edge = links[0]
    assert edge["key"] == "decreases activity of"
    assert edge["source"] == "MESH:D000068877"
    assert edge["target"] == "UniProt:P00519"
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_drugmechdb.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'path_embedding.data'"

**Step 4: Write minimal implementation**

Create directory: `mkdir -p src/path_embedding/data`

Create file `src/path_embedding/data/__init__.py`:
```python
"""Data loading and processing modules."""
```

Create file `src/path_embedding/data/drugmechdb.py`:

```python
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
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_drugmechdb.py -v`

Expected: PASS (all tests)

**Step 6: Run doctests**

Run: `uv run pytest tests/test_drugmechdb.py --doctest-modules`

Expected: PASS

**Step 7: Commit**

```bash
git add src/path_embedding/data/ tests/test_drugmechdb.py tests/data/
git commit -m "feat: add DrugMechDB YAML loader"
```

---

### Task 1.3: NetworkX Multigraph Construction

**Files:**
- Create: `src/path_embedding/utils/__init__.py`
- Create: `src/path_embedding/utils/path_extraction.py`
- Test: `tests/test_path_extraction.py`

**Step 1: Write the failing test**

Create file `tests/test_path_extraction.py`:

```python
"""Tests for path extraction from multigraphs."""
import networkx as nx
from path_embedding.utils.path_extraction import build_multigraph, find_drug_disease_nodes
from path_embedding.data.drugmechdb import load_drugmechdb


def test_build_multigraph():
    """Test building NetworkX multigraph from indication.

    >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    >>> graph = build_multigraph(indications[0])
    >>> isinstance(graph, nx.MultiDiGraph)
    True
    >>> graph.number_of_nodes()
    3
    >>> graph.number_of_edges()
    2
    """
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    indication = indications[0]

    graph = build_multigraph(indication)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_build_multigraph_node_attributes():
    """Test that nodes have correct attributes."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[0])

    # Check drug node
    drug_id = "MESH:D000068877"
    assert drug_id in graph.nodes
    assert graph.nodes[drug_id]["label"] == "Drug"
    assert graph.nodes[drug_id]["name"] == "imatinib"


def test_build_multigraph_edge_attributes():
    """Test that edges have correct attributes."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[0])

    # Check edge
    edges = list(graph.edges(data=True, keys=True))
    source, target, key, data = edges[0]

    assert data["key"] == "decreases activity of"


def test_find_drug_disease_nodes():
    """Test finding drug and disease nodes in graph.

    >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    >>> graph = build_multigraph(indications[0])
    >>> drug_id, disease_id = find_drug_disease_nodes(graph)
    >>> drug_id
    'MESH:D000068877'
    >>> disease_id
    'MESH:D015464'
    """
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[0])

    drug_id, disease_id = find_drug_disease_nodes(graph)

    assert drug_id == "MESH:D000068877"
    assert disease_id == "MESH:D015464"
    assert graph.nodes[drug_id]["label"] == "Drug"
    assert graph.nodes[disease_id]["label"] == "Disease"


def test_find_drug_disease_nodes_second_example():
    """Test with second example (multiple intermediate nodes)."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[1])

    drug_id, disease_id = find_drug_disease_nodes(graph)

    assert graph.nodes[drug_id]["label"] == "Drug"
    assert graph.nodes[disease_id]["label"] == "Disease"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_path_extraction.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'path_embedding.utils'"

**Step 3: Write minimal implementation**

Create directory: `mkdir -p src/path_embedding/utils`

Create file `src/path_embedding/utils/__init__.py`:
```python
"""Utility modules."""
```

Create file `src/path_embedding/utils/path_extraction.py`:

```python
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
    graph = nx.MultiDiGraph()

    # Add nodes with attributes
    for node in indication["nodes"]:
        graph.add_node(
            node["id"],
            label=node["label"],
            name=node["name"]
        )

    # Add edges with attributes
    for link in indication["links"]:
        graph.add_edge(
            link["source"],
            link["target"],
            key=link["key"]
        )

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
    drug_id = None
    disease_id = None

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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_path_extraction.py -v`

Expected: PASS (all tests)

**Step 5: Run doctests**

Run: `uv run pytest tests/test_path_extraction.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/utils/ tests/test_path_extraction.py
git commit -m "feat: add NetworkX multigraph construction"
```

---

### Task 1.4: Extract Paths from Multigraph

**Files:**
- Modify: `src/path_embedding/utils/path_extraction.py`
- Modify: `tests/test_path_extraction.py`

**Step 1: Write the failing test**

Add to `tests/test_path_extraction.py`:

```python
from path_embedding.utils.path_extraction import extract_paths
from path_embedding.datamodel.types import Path


def test_extract_paths_simple():
    """Test extracting paths from simple graph.

    >>> from path_embedding.data.drugmechdb import load_drugmechdb
    >>> indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    >>> graph = build_multigraph(indications[0])
    >>> paths = extract_paths(graph, "DB00619_MESH_D015464_1")
    >>> len(paths)
    1
    >>> isinstance(paths[0], Path)
    True
    """
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    indication = indications[0]
    graph = build_multigraph(indication)

    paths = extract_paths(graph, indication["_id"])

    assert len(paths) == 1
    assert isinstance(paths[0], Path)


def test_extract_paths_structure():
    """Test that extracted path has correct structure."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    indication = indications[0]
    graph = build_multigraph(indication)

    paths = extract_paths(graph, indication["_id"])
    path = paths[0]

    # Should have 3 nodes: Drug -> Protein -> Disease
    assert len(path.nodes) == 3
    # Should have 2 edges
    assert len(path.edges) == 2

    # Check drug and disease
    assert path.drug_id == "MESH:D000068877"
    assert path.disease_id == "MESH:D015464"
    assert path.indication_id == "DB00619_MESH_D015464_1"

    # Check node order
    assert path.nodes[0].label == "Drug"
    assert path.nodes[1].label == "Protein"
    assert path.nodes[2].label == "Disease"


def test_extract_paths_multiple():
    """Test extracting multiple paths."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    indication = indications[1]  # Second example has longer path
    graph = build_multigraph(indication)

    paths = extract_paths(graph, indication["_id"])

    # Should extract at least one path
    assert len(paths) >= 1

    # All paths should start with Drug and end with Disease
    for path in paths:
        assert path.nodes[0].label == "Drug"
        assert path.nodes[-1].label == "Disease"


def test_extract_paths_max_limit():
    """Test limiting number of paths extracted."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    indication = indications[0]
    graph = build_multigraph(indication)

    # Extract with limit
    paths = extract_paths(graph, indication["_id"], max_paths=1)

    assert len(paths) <= 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_path_extraction.py::test_extract_paths_simple -v`

Expected: FAIL with "ImportError: cannot import name 'extract_paths'"

**Step 3: Write minimal implementation**

Add to `src/path_embedding/utils/path_extraction.py`:

```python
from typing import List
from path_embedding.datamodel.types import Node, Edge, Path
import random


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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_path_extraction.py -v`

Expected: PASS (all tests)

**Step 5: Run doctests**

Run: `uv run pytest tests/test_path_extraction.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/utils/path_extraction.py tests/test_path_extraction.py
git commit -m "feat: add path extraction from multigraphs"
```

---

## Phase 2: Text Formatting

### Task 2.1: Path to Text Conversion

**Files:**
- Create: `src/path_embedding/embedding/__init__.py`
- Create: `src/path_embedding/embedding/text_formatter.py`
- Test: `tests/test_text_formatter.py`

**Step 1: Write the failing test**

Create file `tests/test_text_formatter.py`:

```python
"""Tests for path to text conversion."""
from path_embedding.embedding.text_formatter import path_to_text
from path_embedding.datamodel.types import Node, Edge, Path


def test_path_to_text_simple():
    """Test converting simple path to text.

    >>> nodes = [
    ...     Node(id="MESH:D001241", label="Drug", name="aspirin"),
    ...     Node(id="UniProt:P00519", label="Protein", name="COX2"),
    ...     Node(id="MESH:D010146", label="Disease", name="pain")
    ... ]
    >>> edges = [
    ...     Edge(key="inhibits", source="MESH:D001241", target="UniProt:P00519"),
    ...     Edge(key="causes", source="UniProt:P00519", target="MESH:D010146")
    ... ]
    >>> path = Path(nodes=nodes, edges=edges, drug_id="MESH:D001241",
    ...             disease_id="MESH:D010146", indication_id="test")
    >>> text = path_to_text(path)
    >>> text
    'Drug: aspirin | inhibits | Protein: COX2 | causes | Disease: pain'
    """
    nodes = [
        Node(id="MESH:D001241", label="Drug", name="aspirin"),
        Node(id="UniProt:P00519", label="Protein", name="COX2"),
        Node(id="MESH:D010146", label="Disease", name="pain"),
    ]
    edges = [
        Edge(key="inhibits", source="MESH:D001241", target="UniProt:P00519"),
        Edge(key="causes", source="UniProt:P00519", target="MESH:D010146"),
    ]
    path = Path(
        nodes=nodes,
        edges=edges,
        drug_id="MESH:D001241",
        disease_id="MESH:D010146",
        indication_id="test_1"
    )

    text = path_to_text(path)

    expected = "Drug: aspirin | inhibits | Protein: COX2 | causes | Disease: pain"
    assert text == expected


def test_path_to_text_real_example():
    """Test with real DrugMechDB example."""
    from path_embedding.data.drugmechdb import load_drugmechdb
    from path_embedding.utils.path_extraction import build_multigraph, extract_paths

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[0])
    paths = extract_paths(graph, indications[0]["_id"])

    text = path_to_text(paths[0])

    # Should contain all components
    assert "Drug:" in text
    assert "Protein:" in text
    assert "Disease:" in text
    assert "|" in text
    assert "decreases activity of" in text or "causes" in text


def test_path_to_text_format():
    """Test text format structure."""
    nodes = [
        Node(id="A", label="Drug", name="drugA"),
        Node(id="B", label="Gene", name="geneB"),
        Node(id="C", label="Disease", name="diseaseC"),
    ]
    edges = [
        Edge(key="regulates", source="A", target="B"),
        Edge(key="affects", source="B", target="C"),
    ]
    path = Path(
        nodes=nodes, edges=edges,
        drug_id="A", disease_id="C", indication_id="test"
    )

    text = path_to_text(path)

    # Check format: Node | Edge | Node | Edge | Node
    parts = [p.strip() for p in text.split("|")]
    assert len(parts) == 5
    assert parts[0].startswith("Drug:")
    assert parts[2].startswith("Gene:")
    assert parts[4].startswith("Disease:")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_text_formatter.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'path_embedding.embedding'"

**Step 3: Write minimal implementation**

Create directory: `mkdir -p src/path_embedding/embedding`

Create file `src/path_embedding/embedding/__init__.py`:
```python
"""Embedding generation modules."""
```

Create file `src/path_embedding/embedding/text_formatter.py`:

```python
"""Convert paths to structured text representations."""
from path_embedding.datamodel.types import Path


def path_to_text(path: Path) -> str:
    """Convert a Path object to structured text format.

    Format: {label}: {name} | {edge_key} | {label}: {name} | ...

    Args:
        path: Path object to convert

    Returns:
        Structured text representation

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
        >>> path_to_text(path)
        'Drug: aspirin | inhibits | Protein: COX2 | causes | Disease: pain'
    """
    parts = []

    for i, node in enumerate(path.nodes):
        # Add node
        parts.append(f"{node.label}: {node.name}")

        # Add edge if not last node
        if i < len(path.edges):
            parts.append(path.edges[i].key)

    return " | ".join(parts)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_text_formatter.py -v`

Expected: PASS (all tests)

**Step 5: Run doctests**

Run: `uv run pytest tests/test_text_formatter.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/embedding/ tests/test_text_formatter.py
git commit -m "feat: add path to text conversion"
```

---

## Phase 3: Negative Sampling

### Task 3.1: Build Node Inventory

**Files:**
- Create: `src/path_embedding/data/negative_sampling.py`
- Test: `tests/test_negative_sampling.py`

**Step 1: Write the failing test**

Create file `tests/test_negative_sampling.py`:

```python
"""Tests for negative example generation."""
from path_embedding.data.negative_sampling import build_node_inventory
from path_embedding.data.drugmechdb import load_drugmechdb
from path_embedding.utils.path_extraction import build_multigraph, extract_paths


def test_build_node_inventory():
    """Test building node inventory grouped by type and disease.

    >>> from path_embedding.datamodel.types import Node, Path
    >>> nodes1 = [
    ...     Node(id="D1", label="Drug", name="drug1"),
    ...     Node(id="G1", label="Gene", name="gene1"),
    ...     Node(id="DIS1", label="Disease", name="disease1")
    ... ]
    >>> path1 = Path(nodes=nodes1, edges=[], drug_id="D1",
    ...              disease_id="DIS1", indication_id="ind1")
    >>> inventory = build_node_inventory([path1])
    >>> "Gene" in inventory
    True
    """
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")

    # Extract all paths
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)

    # Should have node types as keys
    assert isinstance(inventory, dict)
    assert "Protein" in inventory or "BiologicalProcess" in inventory

    # Each node type should map to disease -> nodes
    for node_type, disease_dict in inventory.items():
        assert isinstance(disease_dict, dict)


def test_build_node_inventory_structure():
    """Test inventory structure: {node_type: {disease_id: [nodes]}}."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)

    # Check structure
    for node_type, disease_dict in inventory.items():
        for disease_id, nodes in disease_dict.items():
            assert isinstance(nodes, list)
            assert len(nodes) > 0
            # All nodes should have the expected type
            for node in nodes:
                assert node.label == node_type


def test_build_node_inventory_excludes_drug_disease():
    """Test that Drug and Disease nodes are excluded from inventory."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)

    # Should not include Drug or Disease in inventory
    assert "Drug" not in inventory
    assert "Disease" not in inventory
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_negative_sampling.py::test_build_node_inventory -v`

Expected: FAIL with "ImportError: cannot import name 'build_node_inventory'"

**Step 3: Write minimal implementation**

Create file `src/path_embedding/data/negative_sampling.py`:

```python
"""Generate negative examples via cross-disease shuffling."""
from typing import List, Dict
from collections import defaultdict
from path_embedding.datamodel.types import Path, Node


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
    inventory = defaultdict(lambda: defaultdict(list))

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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_negative_sampling.py -v`

Expected: PASS (all tests)

**Step 5: Run doctests**

Run: `uv run pytest tests/test_negative_sampling.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/data/negative_sampling.py tests/test_negative_sampling.py
git commit -m "feat: add node inventory builder for negative sampling"
```

---

### Task 3.2: Generate Negative Path

**Files:**
- Modify: `src/path_embedding/data/negative_sampling.py`
- Modify: `tests/test_negative_sampling.py`

**Step 1: Write the failing test**

Add to `tests/test_negative_sampling.py`:

```python
from path_embedding.data.negative_sampling import generate_negative_path
import random


def test_generate_negative_path():
    """Test generating negative path from positive.

    >>> random.seed(42)
    >>> from path_embedding.datamodel.types import Node, Edge, Path
    >>> nodes = [
    ...     Node(id="D1", label="Drug", name="drug1"),
    ...     Node(id="G1", label="Gene", name="gene1"),
    ...     Node(id="DIS1", label="Disease", name="disease1")
    ... ]
    >>> edges = [
    ...     Edge(key="regulates", source="D1", target="G1"),
    ...     Edge(key="causes", source="G1", target="DIS1")
    ... ]
    >>> path = Path(nodes=nodes, edges=edges, drug_id="D1",
    ...             disease_id="DIS1", indication_id="ind1")
    >>> inventory = {"Gene": {"DIS2": [Node(id="G2", label="Gene", name="gene2")]}}
    >>> neg_path = generate_negative_path(path, inventory)
    >>> neg_path.nodes[0].id == "D1"  # Same drug
    True
    >>> neg_path.nodes[-1].id == "DIS1"  # Same disease
    True
    """
    random.seed(42)

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)
    positive_path = all_paths[0]

    negative_path = generate_negative_path(positive_path, inventory)

    # Should preserve drug and disease
    assert negative_path.drug_id == positive_path.drug_id
    assert negative_path.disease_id == positive_path.disease_id

    # Should have same number of nodes
    assert len(negative_path.nodes) == len(positive_path.nodes)

    # First and last nodes should be same
    assert negative_path.nodes[0].id == positive_path.nodes[0].id
    assert negative_path.nodes[-1].id == positive_path.nodes[-1].id


def test_generate_negative_path_preserves_types():
    """Test that negative path preserves node types."""
    random.seed(42)

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)

    # Use second example which has more intermediate nodes
    positive_path = all_paths[1] if len(all_paths) > 1 else all_paths[0]
    negative_path = generate_negative_path(positive_path, inventory)

    # Check that node types match in order
    for i, (pos_node, neg_node) in enumerate(zip(positive_path.nodes, negative_path.nodes)):
        assert pos_node.label == neg_node.label, f"Node {i} type mismatch"


def test_generate_negative_path_different_disease_context():
    """Test that intermediate nodes come from different disease context."""
    random.seed(42)

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")

    # Need at least 2 different diseases
    if len(indications) < 2:
        return  # Skip if not enough data

    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    inventory = build_node_inventory(all_paths)
    positive_path = all_paths[0]

    negative_path = generate_negative_path(positive_path, inventory)

    # At least one intermediate node should be different
    # (if we have enough diversity in the data)
    intermediate_changed = False
    for i in range(1, len(positive_path.nodes) - 1):
        if positive_path.nodes[i].id != negative_path.nodes[i].id:
            intermediate_changed = True
            break

    # If we have multiple diseases, should have changed something
    if len(indications) >= 2:
        assert intermediate_changed
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_negative_sampling.py::test_generate_negative_path -v`

Expected: FAIL with "ImportError: cannot import name 'generate_negative_path'"

**Step 3: Write minimal implementation**

Add to `src/path_embedding/data/negative_sampling.py`:

```python
import random
from path_embedding.datamodel.types import Edge


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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_negative_sampling.py -v`

Expected: PASS (all tests)

**Step 5: Run doctests**

Run: `uv run pytest tests/test_negative_sampling.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/data/negative_sampling.py tests/test_negative_sampling.py
git commit -m "feat: add negative path generation via cross-disease shuffling"
```

---

### Task 3.3: Generate Full Negative Dataset

**Files:**
- Modify: `src/path_embedding/data/negative_sampling.py`
- Modify: `tests/test_negative_sampling.py`

**Step 1: Write the failing test**

Add to `tests/test_negative_sampling.py`:

```python
from path_embedding.data.negative_sampling import generate_negatives


def test_generate_negatives():
    """Test generating negative dataset.

    >>> random.seed(42)
    >>> from path_embedding.datamodel.types import Node, Edge, Path
    >>> nodes = [
    ...     Node(id="D1", label="Drug", name="drug1"),
    ...     Node(id="G1", label="Gene", name="gene1"),
    ...     Node(id="DIS1", label="Disease", name="disease1")
    ... ]
    >>> edges = [Edge(key="reg", source="D1", target="G1"),
    ...          Edge(key="causes", source="G1", target="DIS1")]
    >>> path = Path(nodes=nodes, edges=edges, drug_id="D1",
    ...             disease_id="DIS1", indication_id="ind1")
    >>> negs = generate_negatives([path])
    >>> len(negs) == 1
    True
    """
    random.seed(42)

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    negatives = generate_negatives(all_paths)

    # Should have 1:1 ratio
    assert len(negatives) == len(all_paths)

    # Each negative should be valid Path
    for neg in negatives:
        assert isinstance(neg, Path)
        assert len(neg.nodes) > 0
        assert neg.nodes[0].label == "Drug"
        assert neg.nodes[-1].label == "Disease"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_negative_sampling.py::test_generate_negatives -v`

Expected: FAIL with "ImportError: cannot import name 'generate_negatives'"

**Step 3: Write minimal implementation**

Add to `src/path_embedding/data/negative_sampling.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_negative_sampling.py -v`

Expected: PASS (all tests)

**Step 5: Run doctests**

Run: `uv run pytest tests/test_negative_sampling.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/data/negative_sampling.py tests/test_negative_sampling.py
git commit -m "feat: add full negative dataset generation"
```

---

## Phase 4: OpenAI Embeddings

### Task 4.1: OpenAI API Integration

**Files:**
- Create: `src/path_embedding/embedding/openai_embedder.py`
- Test: `tests/test_openai_embedder.py`

**Step 1: Write the failing test**

Create file `tests/test_openai_embedder.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_embedder.py::test_load_api_key -v`

Expected: FAIL with "ImportError: cannot import name 'load_api_key'"

**Step 3: Write minimal implementation**

Create file `src/path_embedding/embedding/openai_embedder.py`:

```python
"""OpenAI embedding generation."""
import numpy as np
from openai import OpenAI


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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_embedder.py::test_load_api_key -v`

Expected: PASS

Run: `uv run pytest tests/test_openai_embedder.py::test_embed_text -v`

Expected: PASS (makes real API call, may take a few seconds)

**Step 5: Run all embedding tests**

Run: `uv run pytest tests/test_openai_embedder.py -v`

Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/path_embedding/embedding/openai_embedder.py tests/test_openai_embedder.py
git commit -m "feat: add OpenAI embedding generation"
```

---

### Task 4.2: Batch Path Embedding

**Files:**
- Modify: `src/path_embedding/embedding/openai_embedder.py`
- Modify: `tests/test_openai_embedder.py`

**Step 1: Write the failing test**

Add to `tests/test_openai_embedder.py`:

```python
from path_embedding.embedding.openai_embedder import embed_paths
from path_embedding.data.drugmechdb import load_drugmechdb
from path_embedding.utils.path_extraction import build_multigraph, extract_paths


def test_embed_paths():
    """Test embedding multiple paths.

    Note: Makes real API calls, may be slow.
    """
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
    key = load_api_key("/Users/jtr4v/openai.key.another")

    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")
    graph = build_multigraph(indications[0])
    paths = extract_paths(graph, indications[0]["_id"])

    embeddings = embed_paths(paths, key)

    assert embeddings.shape[0] == len(paths)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_embedder.py::test_embed_paths -v`

Expected: FAIL with "ImportError: cannot import name 'embed_paths'"

**Step 3: Write minimal implementation**

Add to `src/path_embedding/embedding/openai_embedder.py`:

```python
from typing import List
from path_embedding.datamodel.types import Path
from path_embedding.embedding.text_formatter import path_to_text


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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_embedder.py::test_embed_paths -v`

Expected: PASS (makes API calls)

**Step 5: Run all tests**

Run: `uv run pytest tests/test_openai_embedder.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/embedding/openai_embedder.py tests/test_openai_embedder.py
git commit -m "feat: add batch path embedding"
```

---

## Phase 5: Train/Test Split

### Task 5.1: Indication-Level Splitting

**Files:**
- Create: `src/path_embedding/model/__init__.py`
- Create: `src/path_embedding/model/data_split.py`
- Test: `tests/test_data_split.py`

**Step 1: Write the failing test**

Create file `tests/test_data_split.py`:

```python
"""Tests for train/test splitting."""
from path_embedding.model.data_split import split_by_indication
from path_embedding.data.drugmechdb import load_drugmechdb
from path_embedding.utils.path_extraction import build_multigraph, extract_paths


def test_split_by_indication():
    """Test splitting paths by indication.

    >>> from path_embedding.datamodel.types import Node, Edge, Path
    >>> paths = [
    ...     Path(nodes=[], edges=[], drug_id="D1", disease_id="DIS1",
    ...          indication_id="ind1"),
    ...     Path(nodes=[], edges=[], drug_id="D1", disease_id="DIS1",
    ...          indication_id="ind1"),
    ...     Path(nodes=[], edges=[], drug_id="D2", disease_id="DIS2",
    ...          indication_id="ind2"),
    ... ]
    >>> train, test = split_by_indication(paths, test_size=0.5, random_seed=42)
    >>> len(train) + len(test) == 3
    True
    """
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")

    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    train_paths, test_paths = split_by_indication(
        all_paths,
        test_size=0.2,
        random_seed=42
    )

    # Should have train and test
    assert len(train_paths) > 0
    assert len(test_paths) > 0

    # Total should equal input
    assert len(train_paths) + len(test_paths) == len(all_paths)


def test_split_by_indication_no_leakage():
    """Test that same indication doesn't appear in both train and test."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")

    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    train_paths, test_paths = split_by_indication(
        all_paths,
        test_size=0.5,
        random_seed=42
    )

    # Get indication IDs from train and test
    train_indications = set(p.indication_id for p in train_paths)
    test_indications = set(p.indication_id for p in test_paths)

    # Should have no overlap
    assert len(train_indications & test_indications) == 0


def test_split_by_indication_reproducible():
    """Test that split is reproducible with same seed."""
    indications = load_drugmechdb("tests/data/sample_drugmechdb.yaml")

    all_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"])
        all_paths.extend(paths)

    train1, test1 = split_by_indication(all_paths, test_size=0.5, random_seed=42)
    train2, test2 = split_by_indication(all_paths, test_size=0.5, random_seed=42)

    # Same seed should give same split
    assert len(train1) == len(train2)
    assert len(test1) == len(test2)

    train_ids_1 = [p.indication_id for p in train1]
    train_ids_2 = [p.indication_id for p in train2]
    assert train_ids_1 == train_ids_2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_split.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'path_embedding.model'"

**Step 3: Write minimal implementation**

Create directory: `mkdir -p src/path_embedding/model`

Create file `src/path_embedding/model/__init__.py`:
```python
"""Model training and evaluation modules."""
```

Create file `src/path_embedding/model/data_split.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data_split.py -v`

Expected: PASS (all tests)

**Step 5: Run doctests**

Run: `uv run pytest tests/test_data_split.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/model/ tests/test_data_split.py
git commit -m "feat: add indication-level train/test split"
```

---

## Phase 6: Classifier Training and Evaluation

### Task 6.1: Random Forest Classifier

**Files:**
- Create: `src/path_embedding/model/classifier.py`
- Test: `tests/test_classifier.py`

**Step 1: Write the failing test**

Create file `tests/test_classifier.py`:

```python
"""Tests for classifier training."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from path_embedding.model.classifier import train_classifier


def test_train_classifier():
    """Test training Random Forest classifier.

    >>> X = np.random.rand(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> model = train_classifier(X, y)
    >>> isinstance(model, RandomForestClassifier)
    True
    """
    # Create synthetic data
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    model = train_classifier(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)

    # Should be able to make predictions
    predictions = model.predict(X_train)
    assert predictions.shape[0] == 100


def test_train_classifier_predictions():
    """Test that trained model can make predictions."""
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)

    model = train_classifier(X_train, y_train)

    X_test = np.random.rand(10, 10)
    predictions = model.predict(X_test)

    assert predictions.shape[0] == 10
    # Predictions should be 0 or 1
    assert all(p in [0, 1] for p in predictions)


def test_train_classifier_probabilities():
    """Test that model can output probabilities."""
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)

    model = train_classifier(X_train, y_train)

    X_test = np.random.rand(10, 10)
    probs = model.predict_proba(X_test)

    assert probs.shape == (10, 2)
    # Probabilities should sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classifier.py -v`

Expected: FAIL with "ImportError: cannot import name 'train_classifier'"

**Step 3: Write minimal implementation**

Create file `src/path_embedding/model/classifier.py`:

```python
"""Classifier training and prediction."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> RandomForestClassifier:
    """Train Random Forest classifier.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        random_state: Random seed for reproducibility

    Returns:
        Trained RandomForestClassifier

    Example:
        >>> X = np.random.rand(50, 10)
        >>> y = np.random.randint(0, 2, 50)
        >>> model = train_classifier(X, y)
        >>> isinstance(model, RandomForestClassifier)
        True
    """
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_classifier.py -v`

Expected: PASS (all tests)

**Step 5: Run doctests**

Run: `uv run pytest tests/test_classifier.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/model/classifier.py tests/test_classifier.py
git commit -m "feat: add Random Forest classifier training"
```

---

### Task 6.2: Evaluation Metrics

**Files:**
- Create: `src/path_embedding/model/evaluation.py`
- Test: `tests/test_evaluation.py`

**Step 1: Write the failing test**

Create file `tests/test_evaluation.py`:

```python
"""Tests for model evaluation."""
import numpy as np
from path_embedding.model.evaluation import evaluate_classifier
from path_embedding.model.classifier import train_classifier


def test_evaluate_classifier():
    """Test evaluating classifier with metrics.

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X = np.random.rand(50, 10)
    >>> y = np.array([0] * 25 + [1] * 25)
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X, y)
    RandomForestClassifier(random_state=42)
    >>> metrics = evaluate_classifier(model, X, y)
    >>> "accuracy" in metrics
    True
    """
    # Create simple dataset
    X_train = np.random.rand(100, 10)
    y_train = np.array([0] * 50 + [1] * 50)

    model = train_classifier(X_train, y_train)

    X_test = np.random.rand(50, 10)
    y_test = np.array([0] * 25 + [1] * 25)

    metrics = evaluate_classifier(model, X_test, y_test)

    # Should have all expected metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    # All metrics should be between 0 and 1
    for metric_name, value in metrics.items():
        assert 0 <= value <= 1, f"{metric_name} out of range"


def test_evaluate_classifier_perfect():
    """Test with perfect predictions."""
    # Create separable data
    X_train = np.vstack([
        np.random.rand(50, 10) - 1,  # Class 0
        np.random.rand(50, 10) + 1,  # Class 1
    ])
    y_train = np.array([0] * 50 + [1] * 50)

    model = train_classifier(X_train, y_train)

    # Test on training data (should be nearly perfect)
    metrics = evaluate_classifier(model, X_train, y_train)

    # Should have high accuracy
    assert metrics["accuracy"] > 0.8
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation.py -v`

Expected: FAIL with "ImportError: cannot import name 'evaluate_classifier'"

**Step 3: Write minimal implementation**

Create file `src/path_embedding/model/evaluation.py`:

```python
"""Model evaluation utilities."""
from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate classifier and return metrics.

    Args:
        model: Trained classifier with predict() and predict_proba()
        X_test: Test features
        y_test: Test labels

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> X = np.random.rand(50, 10)
        >>> y = np.array([0] * 25 + [1] * 25)
        >>> model = RandomForestClassifier(random_state=42).fit(X, y)
        >>> metrics = evaluate_classifier(model, X, y)
        >>> "accuracy" in metrics
        True
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def print_evaluation_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    """Print detailed evaluation report.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    """
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Implausible", "Plausible"]))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n=== Metrics Summary ===")
    metrics = evaluate_classifier(model, X_test, y_test)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation.py -v`

Expected: PASS (all tests)

**Step 5: Run doctests**

Run: `uv run pytest tests/test_evaluation.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/model/evaluation.py tests/test_evaluation.py
git commit -m "feat: add classifier evaluation metrics"
```

---

## Phase 7: CLI Integration

### Task 7.1: Training Command

**Files:**
- Modify: `src/path_embedding/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

Create file `tests/test_cli.py`:

```python
"""Tests for CLI commands."""
import os
import tempfile
from typer.testing import CliRunner
from path_embedding.cli import app

runner = CliRunner()


def test_train_command_help():
    """Test that train command shows help."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout.lower()


def test_train_command_integration():
    """Integration test for train command.

    Note: This makes real API calls and may be slow/expensive.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pkl")

        result = runner.invoke(app, [
            "train",
            "--data", "tests/data/sample_drugmechdb.yaml",
            "--output", model_path,
            "--test-size", "0.5",
            "--max-paths-per-indication", "1",
            "--api-key-path", "/Users/jtr4v/openai.key.another"
        ])

        # Should succeed
        assert result.exit_code == 0

        # Model file should be created
        assert os.path.exists(model_path)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_train_command_help -v`

Expected: FAIL (train command doesn't exist yet)

**Step 3: Write minimal implementation**

Modify `src/path_embedding/cli.py`:

```python
"""CLI interface for path-embedding."""

import typer
from typing_extensions import Annotated
from pathlib import Path
import pickle
import numpy as np

app = typer.Typer(help="path-embedding: Classifier that uses embeddings to find useful paths between drugs and disease")


@app.command()
def train(
    data: Annotated[str, typer.Option(help="Path to DrugMechDB YAML file")],
    output: Annotated[str, typer.Option(help="Path to save trained model (.pkl)")],
    api_key_path: Annotated[str, typer.Option(help="Path to OpenAI API key file")] = "/Users/jtr4v/openai.key.another",
    test_size: Annotated[float, typer.Option(help="Fraction for test set")] = 0.2,
    max_paths_per_indication: Annotated[int, typer.Option(help="Max paths to extract per indication")] = 10,
    random_seed: Annotated[int, typer.Option(help="Random seed")] = 42,
):
    """Train path embedding classifier on DrugMechDB data."""
    from path_embedding.data.drugmechdb import load_drugmechdb
    from path_embedding.utils.path_extraction import build_multigraph, extract_paths
    from path_embedding.data.negative_sampling import generate_negatives
    from path_embedding.model.data_split import split_by_indication
    from path_embedding.embedding.openai_embedder import load_api_key, embed_paths
    from path_embedding.model.classifier import train_classifier
    from path_embedding.model.evaluation import evaluate_classifier, print_evaluation_report

    typer.echo("Loading DrugMechDB data...")
    indications = load_drugmechdb(data)
    typer.echo(f"Loaded {len(indications)} indications")

    typer.echo("Extracting paths from multigraphs...")
    all_positive_paths = []
    for indication in indications:
        graph = build_multigraph(indication)
        paths = extract_paths(graph, indication["_id"], max_paths=max_paths_per_indication)
        all_positive_paths.extend(paths)
    typer.echo(f"Extracted {len(all_positive_paths)} positive paths")

    typer.echo("Generating negative examples...")
    negative_paths = generate_negatives(all_positive_paths)
    typer.echo(f"Generated {len(negative_paths)} negative paths")

    typer.echo("Splitting train/test by indication...")
    train_pos, test_pos = split_by_indication(all_positive_paths, test_size=test_size, random_seed=random_seed)
    train_neg, test_neg = split_by_indication(negative_paths, test_size=test_size, random_seed=random_seed)

    train_paths = train_pos + train_neg
    test_paths = test_pos + test_neg

    train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))
    test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))

    typer.echo(f"Train: {len(train_paths)} paths ({len(train_pos)} pos, {len(train_neg)} neg)")
    typer.echo(f"Test: {len(test_paths)} paths ({len(test_pos)} pos, {len(test_neg)} neg)")

    typer.echo("Loading API key...")
    api_key = load_api_key(api_key_path)

    typer.echo("Generating embeddings for training set...")
    train_embeddings = embed_paths(train_paths, api_key)
    typer.echo(f"Train embeddings shape: {train_embeddings.shape}")

    typer.echo("Generating embeddings for test set...")
    test_embeddings = embed_paths(test_paths, api_key)
    typer.echo(f"Test embeddings shape: {test_embeddings.shape}")

    typer.echo("Training Random Forest classifier...")
    model = train_classifier(train_embeddings, train_labels, random_state=random_seed)

    typer.echo("Evaluating on test set...")
    print_evaluation_report(model, test_embeddings, test_labels)

    typer.echo(f"Saving model to {output}...")
    with open(output, 'wb') as f:
        pickle.dump(model, f)

    typer.echo("Training complete!")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py::test_train_command_help -v`

Expected: PASS

Run full integration test (this will make API calls):
`uv run pytest tests/test_cli.py::test_train_command_integration -v`

Expected: PASS (may take time due to API calls)

**Step 5: Test manually**

Run: `uv run path-embedding train --help`

Expected: Shows help for train command

**Step 6: Commit**

```bash
git add src/path_embedding/cli.py tests/test_cli.py
git commit -m "feat: add train CLI command"
```

---

## Phase 8: KGX Stub

### Task 8.1: KGX Loader Stub

**Files:**
- Create: `src/path_embedding/data/kgx.py`
- Test: `tests/test_kgx.py`

**Step 1: Write the failing test**

Create file `tests/test_kgx.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kgx.py -v`

Expected: FAIL with "ImportError: cannot import name 'load_kgx_paths'"

**Step 3: Write minimal implementation**

Create file `src/path_embedding/data/kgx.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_kgx.py -v`

Expected: PASS

**Step 5: Run doctests**

Run: `uv run pytest tests/test_kgx.py --doctest-modules`

Expected: PASS

**Step 6: Commit**

```bash
git add src/path_embedding/data/kgx.py tests/test_kgx.py
git commit -m "feat: add KGX loader stub for future implementation"
```

---

## Final Steps

### Task 9.1: Run Full Test Suite

**Step 1: Run all tests**

Run: `just test`

Expected: All tests pass, type checking passes, formatting passes

**Step 2: If any failures, fix them**

Address any test failures, type errors, or formatting issues.

**Step 3: Commit fixes**

```bash
git add .
git commit -m "fix: address test/typing/formatting issues"
```

---

### Task 9.2: Create Data and Models Directories

**Step 1: Create directories**

Run:
```bash
mkdir -p data models
echo "# DrugMechDB data files" > data/README.md
echo "# Trained models" > models/README.md
```

**Step 2: Add to git**

```bash
git add data/README.md models/README.md
git commit -m "chore: add data and models directories"
```

---

### Task 9.3: Update Documentation

**Step 1: Verify README is up to date**

Check that `README.md` matches what was written earlier.

**Step 2: Verify design doc is complete**

Check `docs/plans/2025-11-10-path-embedding-classifier-design.md`

**Step 3: Add usage examples to docs**

Create `docs/usage.md` with examples of how to use the CLI.

**Step 4: Commit**

```bash
git add docs/
git commit -m "docs: add usage examples"
```

---

## Success Criteria

After completing all tasks:

- [ ] All tests pass: `just test`
- [ ] Type checking passes: `just mypy`
- [ ] Formatting passes: `just format`
- [ ] Can train model: `uv run path-embedding train --data <file> --output model.pkl`
- [ ] Model saves successfully
- [ ] Evaluation metrics printed
- [ ] README updated
- [ ] Design document complete
- [ ] All commits made with clear messages

---

## Notes

- **API Costs**: The embedding generation makes real OpenAI API calls. Use small test datasets during development to minimize costs.
- **Test Data**: The `tests/data/sample_drugmechdb.yaml` fixture contains only 2 indications. For real training, download the full DrugMechDB YAML file.
- **Random Seeds**: All random operations use seeds for reproducibility (42 default).
- **Future Work**: KGX support, node enrichment, alternative classifiers are noted but not implemented.

---

## Execution

**Plan complete and saved to `docs/plans/2025-11-10-path-embedding-implementation.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
