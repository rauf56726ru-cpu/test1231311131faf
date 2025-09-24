"""Ingest Notion export into local memory store."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..config import get_settings
from ..utils.logging import get_logger
from .embedder import ImageEmbedder, TextEmbedder
from .faiss_store import FaissStore
from .retriever import MemoryRetriever
from .vision_ocr import extract_text

LOGGER = get_logger(__name__)


@dataclass
class NotionDocument:
    path: Path
    title: str
    content: str
    tags: List[str]
    kind: str  # text | image


def _parse_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _parse_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return ""


def load_export(path: Path) -> List[NotionDocument]:
    documents: List[NotionDocument] = []
    for file in path.rglob("*"):
        if file.is_dir():
            continue
        suffix = file.suffix.lower()
        title = file.stem.replace("-", " ")
        tags: List[str] = []
        if suffix in {".md", ".txt", ".html"}:
            content = _parse_text(file)
            documents.append(NotionDocument(path=file, title=title, content=content, tags=tags, kind="text"))
        elif suffix in {".json"}:
            content = _parse_json(file)
            documents.append(NotionDocument(path=file, title=title, content=content, tags=tags, kind="text"))
        elif suffix in {".png", ".jpg", ".jpeg", ".gif"}:
            text = extract_text(file)
            documents.append(NotionDocument(path=file, title=title, content=text, tags=tags, kind="image"))
    return documents


def ingest_notion(path: Path | str, memory_dir: Path | str = Path("data/memory")) -> MemoryRetriever:
    path = Path(path)
    memory_dir = Path(memory_dir)
    settings = get_settings()
    store = FaissStore(memory_dir)
    store.load(text_dim=settings.memory.faiss_dim_text, image_dim=settings.memory.faiss_dim_image)

    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()

    docs = load_export(path)
    text_docs = [doc for doc in docs if doc.kind == "text" and doc.content.strip()]
    if text_docs:
        vectors = text_embedder.embed([doc.content for doc in text_docs])
        metadatas = [
            {
                "title": doc.title,
                "path": str(doc.path),
                "tags": doc.tags,
                "kind": doc.kind,
            }
            for doc in text_docs
        ]
        store.add_text(vectors, metadatas)
        LOGGER.info("Indexed %s text documents", len(text_docs))

    image_docs = [doc for doc in docs if doc.kind == "image"]
    if image_docs:
        vectors = image_embedder.embed([doc.path for doc in image_docs])
        metadatas = [
            {
                "title": doc.title,
                "path": str(doc.path),
                "tags": doc.tags,
                "kind": doc.kind,
                "ocr": doc.content,
            }
            for doc in image_docs
        ]
        store.add_images(vectors, metadatas)
        LOGGER.info("Indexed %s images", len(image_docs))

    return MemoryRetriever(store=store, text_embedder=text_embedder, image_embedder=image_embedder)


if __name__ == "__main__":
    ingest_notion(Path("./notion_export"))
