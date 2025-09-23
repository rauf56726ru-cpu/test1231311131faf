"""Embedding utilities for text and images."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional heavy dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover
    import torch
    from transformers import CLIPProcessor, CLIPModel
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    CLIPProcessor = None  # type: ignore
    CLIPModel = None  # type: ignore


@dataclass
class TextEmbedder:
    model_name: str = "all-MiniLM-L6-v2"
    allowed_models: Optional[Iterable[str]] = None

    def __post_init__(self) -> None:
        self.allowed_models = list(self.allowed_models or [self.model_name])
        self._model = None
        self._model_id: Optional[str] = None
        self._dim = 384
        self.set_text_model(self.model_name, _initial=True)

    def current_name(self) -> str:
        return self.model_name

    def set_text_model(self, name: str, _initial: bool = False) -> None:
        if name not in self.allowed_models and self.allowed_models:
            raise ValueError(f"Text model '{name}' is not allowed")
        if not _initial and name == self.model_name and self._model is not None:
            return
        self.model_name = name
        identifier = self._resolve_model_identifier(name)
        self._model_id = identifier
        self._model = None
        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(identifier)
                if hasattr(self._model, "get_sentence_embedding_dimension"):
                    self._dim = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                self._model = None
        if self._model is None:
            self._dim = 384

    def allowed(self) -> List[str]:
        return list(self.allowed_models or [])

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if self._model is not None:
            return np.array(self._model.encode(texts, show_progress_bar=False))
        return np.array([self._hash_embedding(text, self._dim) for text in texts])

    @staticmethod
    def _resolve_model_identifier(name: str) -> str:
        if "/" in name:
            return name
        return f"sentence-transformers/{name}"

    @staticmethod
    def _hash_embedding(text: str, dim: int = 384) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeat = (dim + len(digest) - 1) // len(digest)
        data = (digest * repeat)[:dim]
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        norm = np.linalg.norm(arr)
        return arr / norm if norm else arr


@dataclass
class ImageEmbedder:
    model_name: str = "openai/clip-vit-base-patch32"

    def __post_init__(self) -> None:
        self.model = None
        self.processor = None
        if CLIPModel is not None:
            try:
                self.model = CLIPModel.from_pretrained(self.model_name)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
            except Exception:
                self.model = None
                self.processor = None

    def embed(self, image_paths: Iterable[Path]) -> np.ndarray:
        paths = list(image_paths)
        if self.model is None or self.processor is None or Image is None:
            return np.array([self._hash_embedding(path) for path in paths])
        images = [Image.open(path).convert("RGB") for path in paths]
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        embeddings = embeddings.detach().cpu().numpy()
        return embeddings

    @staticmethod
    def _hash_embedding(path: Path, dim: int = 512) -> np.ndarray:
        digest = hashlib.sha256(str(path).encode("utf-8")).digest()
        repeat = (dim + len(digest) - 1) // len(digest)
        data = (digest * repeat)[:dim]
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        return arr / np.linalg.norm(arr) if np.linalg.norm(arr) else arr
