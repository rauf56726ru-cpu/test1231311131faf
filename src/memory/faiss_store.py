"""Local FAISS-backed vector store with graceful fallbacks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    LOGGER.info("FAISS unavailable (%s); falling back to numpy store", exc)
    faiss = None  # type: ignore

META_TEXT = "text_meta.json"
META_IMAGE = "image_meta.json"


@dataclass
class VectorIndex:
    dim: int
    kind: str
    index: Optional[object] = None
    metadata: List[Dict[str, object]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = []
        if not hasattr(self, "_vectors"):
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, object]]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if faiss is not None:
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(vectors)
        self._vectors = np.vstack([self._vectors, vectors]) if len(self._vectors) else vectors
        self.metadata.extend(metadatas)

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, object]]]:
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        if faiss is not None and self.index is not None:
            D, I = self.index.search(query, min(k, len(self.metadata)))
            out: List[Tuple[float, Dict[str, object]]] = []
            for d, i in zip(D[0], I[0]):
                if 0 <= i < len(self.metadata):
                    out.append((float(d), self.metadata[i]))
            return out
        # fallback cosine-like через L2 на нормированных векторах
        if len(self.metadata) == 0 or not hasattr(self, "_vectors") or len(self._vectors) == 0:
            return []
        q = query[0]
        # нормализация
        A = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-8)
        qn = q / (np.linalg.norm(q) + 1e-8)
        sims = A @ qn
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.metadata[i]) for i in idx]

    # Только для fallback JSON
    def serialize(self) -> Dict[str, object]:
        if faiss is not None and getattr(self, "index", None) is not None and hasattr(self.index, "xb"):
            vectors = faiss.vector_to_array(self.index.xb).reshape(-1, self.dim)
        else:
            vectors = self._vectors
        return {
            "dim": self.dim,
            "kind": self.kind,
            "metadata": self.metadata,
            "vectors": vectors.tolist(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, object]) -> "VectorIndex":
        index = cls(dim=int(data["dim"]), kind=str(data.get("kind", "text")))
        vectors = np.array(data.get("vectors", []), dtype=np.float32)
        metadata = data.get("metadata", [])
        if len(vectors):
            index.add(vectors, list(metadata))
        else:
            index.metadata = list(metadata)
        return index


class FaissStore:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.text_index: Optional[VectorIndex] = None
        self.image_index: Optional[VectorIndex] = None

    def load(self, text_dim: int, image_dim: int) -> None:
        # попытка бинарных индексов
        t_idx = self.base_path / "text.index"
        i_idx = self.base_path / "image.index"
        t_meta = self.base_path / META_TEXT
        i_meta = self.base_path / META_IMAGE

        if faiss is not None and t_idx.exists():
            vi = VectorIndex(dim=text_dim, kind="text")
            vi.index = faiss.read_index(str(t_idx))
            vi.metadata = json.loads(t_meta.read_text()) if t_meta.exists() else []
            if hasattr(vi.index, "xb"):
                vectors = faiss.vector_to_array(vi.index.xb).reshape(-1, text_dim)
                vi._vectors = vectors
            self.text_index = vi
        else:
            t_json = self.base_path / "text_index.json"
            self.text_index = VectorIndex.deserialize(json.loads(t_json.read_text())) if t_json.exists() else VectorIndex(text_dim, "text")

        if faiss is not None and i_idx.exists():
            vi = VectorIndex(dim=image_dim, kind="image")
            vi.index = faiss.read_index(str(i_idx))
            vi.metadata = json.loads(i_meta.read_text()) if i_meta.exists() else []
            if hasattr(vi.index, "xb"):
                vectors = faiss.vector_to_array(vi.index.xb).reshape(-1, image_dim)
                vi._vectors = vectors
            self.image_index = vi
        else:
            i_json = self.base_path / "image_index.json"
            self.image_index = VectorIndex.deserialize(json.loads(i_json.read_text())) if i_json.exists() else VectorIndex(image_dim, "image")

    def persist(self) -> None:
        # текст
        if self.text_index is not None:
            if faiss is not None and getattr(self.text_index, "index", None) is not None:
                faiss.write_index(self.text_index.index, str(self.base_path / "text.index"))
                (self.base_path / META_TEXT).write_text(json.dumps(self.text_index.metadata))
            (self.base_path / "text_index.json").write_text(json.dumps(self.text_index.serialize()))
        # изображения
        if self.image_index is not None:
            if faiss is not None and getattr(self.image_index, "index", None) is not None:
                faiss.write_index(self.image_index.index, str(self.base_path / "image.index"))
                (self.base_path / META_IMAGE).write_text(json.dumps(self.image_index.metadata))
            (self.base_path / "image_index.json").write_text(json.dumps(self.image_index.serialize()))

    def add_text(self, vectors: np.ndarray, metadatas: List[Dict[str, object]]) -> None:
        if self.text_index is None:
            raise RuntimeError("Text index not initialised")
        self.text_index.add(vectors, metadatas)
        self.persist()

    def add_images(self, vectors: np.ndarray, metadatas: List[Dict[str, object]]) -> None:
        if self.image_index is None:
            raise RuntimeError("Image index not initialised")
        self.image_index.add(vectors, metadatas)
        self.persist()

    # Эти методы нужны API/ретриверу
    def search_text(self, query_vector: np.ndarray, k: int = 5):
        if self.text_index is None:
            return []
        return self.text_index.search(query_vector, k=k)

    def search_image(self, query_vector: np.ndarray, k: int = 5):
        if self.image_index is None:
            return []
        return self.image_index.search(query_vector, k=k)
