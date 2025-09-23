"""OCR utilities for Notion attachments."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:  # pragma: no cover
    import pytesseract
    from PIL import Image
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
    Image = None  # type: ignore


def ocr_image(path: Path) -> str:
    if pytesseract is None or Image is None:
        return ""
    try:
        with Image.open(path) as img:
            return pytesseract.image_to_string(img)
    except Exception:
        return ""


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif"}:
        return ocr_image(path)
    return ""
