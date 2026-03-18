# Affective-RAG

[![PyPI version](https://img.shields.io/pypi/v/affective-rag.svg)](https://pypi.org/project/affective-rag/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/jeje1197/AffectiveRAG?style=social)](https://github.com/jeje1197/AffectiveRAG)

A lightweight Python library for emotion-aware episodic memory retrieval using **Affective Link Scoring (ALS)** and **Context Path Traversal (CPT)**.

## Installation

```bash
# Install from PyPI
pip install affective-rag

# Install from source
git clone https://github.com/jeje1197/AffectiveRAG.git
cd AffectiveRAG
pip install .
```

## What is Affective-RAG?

Affective-RAG helps you build systems that remember and retrieve memories based on semantic similarity, emotional resonance, and temporal proximity. Instead of just finding similar text, it find emotionally and contextually relevant memory chains by integrating semantic similarity, emotional similarity, and temporal proximity to move beyond a standard 1D top-k search.

**Key Features:**

- **Memory Graph**: NetworkX-backed directed graph for storing events and relationships
- **Smart Retrieval**: Context Path Traversal algorithm that follows memory chains
- **Affective Scoring**: ALS combines semantic, emotional, and temporal similarity
- **Pluggable**: Bring your own vector database (ChromaDB, FAISS, Pinecone, etc.)
- **Observable**: Built-in logging to understand retrieval decisions
- **Minimal Dependencies** (core library): Just NetworkX, NumPy, and Pydantic

## Quick Start

### Basic Example

```python
from affective_rag import MemoryGraph, CPTConfig

# Create a memory graph
graph = MemoryGraph()

# Add events with emotional and temporal metadata
graph.add_event("e1", {
    "id": "e1",
    "text": "Alice arrives at the party",
    "timestamp": "2024-01-01T18:00:00Z",
    "emotion": {"valence": 0.8, "arousal": 0.6},
    "semantic_vec": [0.1, 0.2, 0.3],  # from your embedding model
    "emotional_vec": [0.8, 0.1, 0.1],
})

graph.add_event("e2", {
    "id": "e2", 
    "text": "Alice argues with Bob",
    "timestamp": "2024-01-01T19:30:00Z",
    "emotion": {"valence": 0.2, "arousal": 0.8},
    "semantic_vec": [0.15, 0.25, 0.28],
    "emotional_vec": [0.1, 0.1, 0.8],
})

# Create temporal/causal links
graph.add_edge("e1", "e2")

# Provide a vector lookup function (connects to your vector DB)
def vector_lookup(query):
    # Your vector DB query here - returns (id, score) tuples
    return [("e1", 0.95), ("e2", 0.75)]

# Retrieve context paths
result = graph.retrieve(
    query="What happened at the party?",
    vector_lookup=vector_lookup,
    cpt_config=CPTConfig(max_depth=3, seed_nodes=3)
)

# Use the results
for path in result.paths:
    print(f"Path score: {path.score:.4f}")
    for node in path.nodes:
        print(f"  - {node['text']}")
```

## Production Pattern: Separate Storage

In production, you typically don't want to duplicate event data. Use the **event_data_provider** pattern:

```python
from affective_rag import MemoryGraph, CPTConfig

# Store only structure (IDs + edges) in MemoryGraph
graph = MemoryGraph()
graph.add_event("e1", {})  # Just the ID
graph.add_event("e2", {})
graph.add_edge("e1", "e2")

# Your event data lives in your database
event_database = {
    "e1": {"id": "e1", "text": "...", "timestamp": "...", ...},
    "e2": {"id": "e2", "text": "...", "timestamp": "...", ...},
}

# Provide a data fetcher
def get_event_data(event_id: str) -> dict:
    return event_database[event_id]

# Retrieve with zero data duplication
result = graph.retrieve(
    query="...",
    vector_lookup=my_vector_lookup,
    event_data_provider=get_event_data,
)
```

The library fetches event data on-demand during traversal, keeping your memory graph lightweight.

## ChromaDB Integration

```python
from affective_rag import MemoryGraph, CPTConfig
import chromadb

# Setup ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection("memories")

# Add your events
collection.add(
    ids=["e1", "e2", "e3"],
    documents=["text1", "text2", "text3"],
)

# Create vector lookup adapter
def chroma_lookup(query, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    ids = results["ids"][0]
    distances = results["distances"][0]
    # Convert distances to similarity scores
    return [(id, 1.0 / (1.0 + dist)) for id, dist in zip(ids, distances)]

# Build your graph and retrieve
graph = MemoryGraph()
# ... add events and edges ...

result = graph.retrieve(
    query="What was discussed?",
    vector_lookup=chroma_lookup,
    cpt_config=CPTConfig(max_depth=3),
)
```

## Configuration

All CPT behavior is controlled via `CPTConfig`:

```python
from affective_rag import CPTConfig

config = CPTConfig(
    seed_nodes=3,           # Max number of starting points from vector search
    max_depth=3,            # Max hops to traverse from each seed
    max_neighbors=10,       # Max candidates to consider at each hop
    min_als_score=0.5,      # Filter out weak connections
    cache_event_data=True,  # Cache event lookups during traversal
)

result = graph.retrieve(query, vector_lookup=lookup, cpt_config=config)
```

## Custom Scoring

Override the default ALS scoring with your own function:

```python
def custom_scorer(event_a: dict, event_b: dict) -> float:
    # Your custom scoring logic
    return some_score

result = graph.retrieve(
    query="...",
    vector_lookup=lookup,
    score_fn=custom_scorer,
)
```

## Observability

Enable logging to see exactly what the algorithm is doing:

```python
import json

def logger(log_data: dict):
    print(json.dumps(log_data, indent=2))

graph.set_logging(enabled=True, logger_provider=logger)

# Now retrieval operations will log:
# - Seed nodes selected
# - Each traversal step
# - Neighbor scores and selections
# - Final context paths
result = graph.retrieve(...)
```

## API Reference

### `MemoryGraph`

**Methods:**
- `add_event(event_id: str, event_data: dict)` - Add/update a memory node
- `add_edge(from_event: str, to_event: str, **attrs)` - Create a directed edge
- `retrieve(query, vector_lookup, cpt_config=None, event_data_provider=None, **kwargs)` - Run CPT
- `set_logging(enabled: bool, logger_provider=None)` - Enable observability
- `get_event(event_id: str) -> dict` - Retrieve node data
- `neighbors(event_id: str) -> List[str]` - Get successors
- `has_node(event_id: str) -> bool` - Check if node exists
- `nodes() -> List[str]` - All node IDs
- `edges() -> List[Tuple]` - All edges

### `CPTConfig`

```python
class CPTConfig:
    seed_nodes: int = 3              # Starting points
    max_depth: int = 3               # Traversal depth
    max_neighbors: int = 10          # Candidates per hop
    min_als_score: Optional[float]   # Score threshold
    cache_event_data: bool = True    # Enable caching
    spreading_activation: bool = False  # Enable/disable semantic spreading
```

### `CPTResult`

```python
class CPTResult:
    paths: List[ContextPath]  # Ranked by score (descending)

class ContextPath:
    nodes: List[dict]  # Full event data for each node in path
    score: float       # Cumulative path score
```

## Affective Link Scoring (ALS)

ALS combines three factors to score memory connections:

1. **Semantic Similarity**: Cosine similarity of semantic embeddings
2. **Emotional Similarity**: Cosine similarity of emotional embeddings
3. **Temporal Proximity**: `1 / (1 + days_apart)`

**Default weights** (from research):
```python
semantic_weight = 0.0791
emotional_weight = -0.5179
temporal_weight = 3.1470
bias = 0.0
```

The algorithm gracefully handles missing vectors or timestamps (treats contribution as 0).

## How Context Path Traversal Works

1. **Seed Selection**: Query your vector DB to find top-k starting nodes
2. **Greedy Traversal**: From each seed, greedily follow the highest-scoring edge
3. **Avoid Cycles**: Track visited nodes to prevent loops
4. **Depth Limit**: Stop after `max_depth` hops or when no neighbors remain
5. **Ranking**: Sort all discovered paths by cumulative score

Each traversal step:
- Scores all neighbor candidates using ALS (or custom scorer)
- Filters by `min_als_score` if configured
- Considers top `max_neighbors` candidates
- Selects the single best neighbor (greedy)

## Examples

See the [examples/](examples) directory for complete demos:
- [spreading_demo.py](examples/spreading_demo.py) - Spreading activation discovering unconnected events
- [no_spread_demo.py](examples/no_spread_demo.py) - Graph-only traversal comparison
- [topk_demo.py](examples/topk_demo.py) - Top-k sampling configuration
- [unified_sampling_demo.py](examples/unified_sampling_demo.py) - Comprehensive sampling modes
- [logging_example.py](examples/logging_example.py) - Observability with JSON logging

## Documentation

- **[Unified Sampling Guide](docs/UNIFIED_SAMPLING.md)** - Advanced sampling strategies (top-k, top-p, random)
- **[API Reference](#api-reference)** - Complete method documentation
- **[Examples](examples/)** - Working code samples

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Background & Citation

This library implements the retrieval logic described in the paper: **"Beyond Semantics: Information Retrieval with Emotion and Time"**.

If you use this library in your research, please cite:

```bibtex
@misc{evans2025affectiverag,
  author = {Evans, Joseph},
  title = {Beyond Semantics: Information Retrieval with Emotion and Time},
  year = {2025}
}
```

## License & Patents

This project is licensed under the **Apache License 2.0**.

The ALS and CPT algorithms are subject to U.S. Provisional Patent No. 63/949,166. See the [PATENTS](PATENTS) file for usage guidelines including free use for research and personal projects.

Questions? Contact: focalways99@gmail.com
