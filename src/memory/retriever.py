"""Retriever combining embeddings and FAISS store."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


from ..config import get_settings
from .embedder import ImageEmbedder, TextEmbedder
from .faiss_store import FaissStore


@dataclass
class MemoryRetriever:
    store: FaissStore
    text_embedder: TextEmbedder
    image_embedder: ImageEmbedder

    def search_text(self, query: str, topk: int = 5) -> List[Dict[str, object]]:
        if not query:
            return []
        vector = self.text_embedder.embed([query])[0]
        results = self.store.search_text(vector, k=topk)
        enriched = []
        for score, meta in results:
            enriched.append({**meta, "score": score})
        return enriched

    def search_image(self, query_text: str, topk: int = 5) -> List[Dict[str, object]]:
        # simple heuristic: embed query text with text embedder as surrogate
        vector = self.text_embedder.embed([query_text])[0]
        results = self.store.search_image(vector, k=topk)
        return [{**meta, "score": score} for score, meta in results]
