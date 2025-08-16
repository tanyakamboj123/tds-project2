"""
Image utilities:
- Detect if a path is an image (Pillow-based; no imghdr).
- Read bytes + MIME.
- Extract compact metadata (size/format/EXIF).
- Optional OCR text.
- Build a short text summary for LLM fallback.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple, Optional
import base64
from io import BytesIO
import easyocr

# Pillow for detection/metadata/thumbnail
try:
    from PIL import Image, ExifTags
except Exception:
    Image = None
    ExifTags = None



IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


# ---------- Core detection ----------

def is_image_file(path: str) -> bool:
    """True if path is an image we can open with Pillow (or by extension)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return True
    if Image is None:
        return False
    try:
        with Image.open(path) as img:
            img.verify()  # quick integrity check
        return True
    except Exception:
        return False


def pillow_mime(path: str) -> str:
    """Best-effort MIME from Pillow; fallback to extension."""
    if Image is None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        return {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "bmp": "image/bmp",
            "tif": "image/tiff",
            "tiff": "image/tiff",
        }.get(ext, "image/jpeg")
    try:
        with Image.open(path) as img:
            fmt = (img.format or "jpeg").lower()
        return {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "webp": "image/webp",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "tif": "image/tiff",
        }.get(fmt, f"image/{fmt}")
    except Exception:
        return "image/jpeg"


def read_image_bytes(path: str) -> Tuple[bytes, str]:
    with open(path, "rb") as f:
        b = f.read()
    return b, pillow_mime(path)


# ---------- Metadata / OCR / Thumbnails ----------

def image_metadata(path: str) -> Dict[str, Any]:
    """Compact metadata (format/mode/size + selected EXIF)."""
    meta: Dict[str, Any] = {}
    if Image is None:
        return meta
    try:
        with Image.open(path) as img:
            meta["format"] = img.format
            meta["mode"] = img.mode
            meta["size"] = {"width": img.width, "height": img.height}
            # EXIF (common tags only)
            exif_info = {}
            if hasattr(img, "_getexif") and img._getexif():
                for k, v in img._getexif().items():
                    tag = ExifTags.TAGS.get(k, str(k))
                    exif_info[tag] = v
            if exif_info:
                keep = ["DateTime", "Model", "Make", "LensModel", "FNumber", "ExposureTime", "ISOSpeedRatings"]
                meta["exif"] = {k: exif_info[k] for k in keep if k in exif_info}
    except Exception:
        pass
    return meta


##def image_ocr_text(path: str, max_chars: int = 4000) -> str:
    
  ##|     return ""
    #try:
   #     with Image.open(path) as img:
   #         txt = pytesseract.image_to_string(img) or ""
   #         return txt[:max_chars]
   # except Exception:
   #     return 
def image_ocr_text(path: str) -> str:
    reader = easyocr.Reader(['en'])  # loads English model
    results = reader.readtext(path, detail=0)  # detail=0 → just text
    return "\n".join(results)


def image_thumbnail_data_uri(path: str, max_side: int = 512, fmt: str = "PNG") -> Optional[str]:
    """
    Create a small thumbnail data URI (useful for debug UIs / logs).
    Returns None if Pillow unavailable.
    """
    if Image is None:
        return None
    try: 
        with Image.open(path) as img:
            img.thumbnail((max_side, max_side))
            buf = BytesIO()
            img.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
            return f"data:{mime};base64,{b64}"
    except Exception:
        return None


# ---------- Summaries for LLM fallback ----------

def summarize_image_for_prompt(path: str, include_ocr: bool = True) -> str:
    """
    Small textual summary to ground the LLM when a vision model isn’t used.
    Includes size/format/EXIF excerpt + OCR excerpt (optional).
    """
    parts: List[str] = []
    md = image_metadata(path)
    if md:
        parts.append(f"[IMAGE META] {md}")
    if include_ocr:
        ocr = image_ocr_text(path)
        if ocr.strip():
            parts.append("---- OCR (first ~500 chars) ----\n" + ocr[:500])
    return "\n".join(parts) if parts else "[IMAGE: no metadata/OCR available]"


# ---------- High-level loader for images ----------

def load_image_for_context(path: str) -> Dict[str, Any]:
    """
    Returns a standardized dict you can merge into your app’s flow:
      {
        "text": "...meta/ocr text...",      # for text-only model fallback
        "artifacts": {
            "image": {"bytes": <bytes>, "mime": "image/png", "path": "<path>"},
            "thumb": "data:image/png;base64,..."  # optional thumbnail
        }
      }
    """
    out: Dict[str, Any] = {"text": "", "artifacts": {}}
    if not is_image_file(path):
        return out
    img_bytes, mime = read_image_bytes(path)
    out["artifacts"]["image"] = {"bytes": img_bytes, "mime": mime, "path": path}
    thumb = image_thumbnail_data_uri(path)
    if thumb:
        out["artifacts"]["thumb"] = thumb
    out["text"] = summarize_image_for_prompt(path)
    return out
