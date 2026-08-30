"""Retrieval and indexing layer (Checkpoint 3.1, Architecture v6.7 slide 1).

`indexes` is not re-exported here: importing it pulls in the vector store, and the
chunking, ingestion, and computation layers have no need of one.
"""

from cdd_agent.retrieval.chunking import Chunk, SourceDocument, chunk_all, chunk_document
from cdd_agent.retrieval.ingestion import (
    IngestionReport,
    StructuredTable,
    ingest_directory,
    ingest_knowledge_base,
)

__all__ = [
    "Chunk", "IngestionReport", "SourceDocument", "StructuredTable",
    "chunk_all", "chunk_document", "ingest_directory", "ingest_knowledge_base",
]
