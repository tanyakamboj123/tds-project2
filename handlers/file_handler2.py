import os
import re
import io
import zipfile
import tarfile
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from handlers.image_handler import is_image_file, load_image_for_context

import pandas as pd

# Optional imports
try:
    import duckdb
except Exception:
    duckdb = None

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

# PDF helpers
try:
    import pdfplumber  # text + simple table extraction
except Exception:
    pdfplumber = None

try:
    import camelot  # pip install camelot-py[cv]  (needs Ghostscript for lattice on Windows)
except Exception:
    camelot = None


# =========================
# Public API
# =========================

def load_file(file_path: str) -> Dict[str, Any]:
    """
    Universal loader.
    Returns:
      {
        "dataframes": List[pd.DataFrame],  # may be empty
        "text": str or "",                 # may be empty
        "artifacts": Dict[str, Any]        # optional debug info
      }
    """
    ext = os.path.splitext(file_path)[1].lower()
    result: Dict[str, Any] = {"dataframes": [], "text": "", "artifacts": {}}

    try:
        if ext in (".csv", ".tsv"):
            df = _read_delimited(file_path, sep="\t" if ext == ".tsv" else None)
            result["dataframes"].append(df)

        elif ext in (".xlsx", ".xls"):
            result["dataframes"].extend(_read_excel(file_path))

        elif ext == ".parquet":
            if pq is None:
                raise RuntimeError("pyarrow is not installed; cannot read parquet.")
            result["dataframes"].append(pq.read_table(file_path).to_pandas())

        elif ext in (".db", ".duckdb"):
            if duckdb is None:
                raise RuntimeError("duckdb is not installed; cannot read duckdb.")
            result["dataframes"].extend(_read_duckdb(file_path))

        elif ext == ".zip":
            result["dataframes"].extend(_read_zip(file_path))

        elif ext in (".tar", ".gz", ".tgz", ".tar.gz"):
            result["dataframes"].extend(_read_tar(file_path))

        elif ext == ".pdf":
            # Always try both: tables + text
            pdf_tables = extract_pdf_tables(file_path)
            pdf_text = extract_pdf_text(file_path)
            if pdf_tables:
                result["dataframes"].extend(pdf_tables)
            if pdf_text:
                result["text"] = pdf_text
                
        elif is_image_file(file_path):
            img_ctx = load_image_for_context(file_path)
            result["text"] = img_ctx.get("text", "")
            result.setdefault("artifacts", {}).update(img_ctx.get("artifacts", {}))

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return result

    except Exception as e:
        raise ValueError(f"Failed to load file {file_path}: {str(e)}")


def summarize_context(load_result: Dict[str, Any], max_text_chars: int = 2000) -> str:
    """
    Build a compact summary string for the LLM prompt, combining:
      - DataFrames (shapes/columns/small sample)
      - Text excerpt (for PDFs & other text inputs)
    """
    dfs: List[pd.DataFrame] = load_result.get("dataframes", [])
    text: str = load_result.get("text", "") or ""

    parts: List[str] = []

    # Tables
    if dfs:
        parts.append(_summarize_dataframes(dfs))
    else:
        parts.append("No tabular data extracted.")

    # Text
    if text.strip():
        excerpt = text[:max_text_chars]
        parts.append("---- TEXT EXCERPT ----\n" + excerpt)

    return "\n\n".join(p for p in parts if p.strip())


# =========================
# Readers
# =========================

def _read_delimited(file_path: str, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Robust CSV/TSV reader.
      - sep=None triggers pandas dialect sniffing (python engine)
      - on_bad_lines='skip' to survive messy rows
    """
    try:
        if sep is None:
            # auto-detect
            return pd.read_csv(file_path, sep=None, engine="python", on_bad_lines="skip")
        else:
            return pd.read_csv(file_path, sep=sep, on_bad_lines="skip")
    except Exception:
        # final fallback
        return pd.read_csv(file_path, on_bad_lines="skip")


def _read_excel(file_path: str) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    xls = pd.ExcelFile(file_path)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        dfs.append(df)
    return dfs


def _read_duckdb(file_path: str) -> List[pd.DataFrame]:
    con = duckdb.connect(file_path)
    try:
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        out: List[pd.DataFrame] = []
        for t in tables:
            out.append(con.execute(f"SELECT * FROM {t}").fetchdf())
        return out
    finally:
        con.close()


def _read_zip(file_path: str) -> List[pd.DataFrame]:
    tmp_dir = tempfile.mkdtemp()
    out: List[pd.DataFrame] = []
    with zipfile.ZipFile(file_path, "r") as z:
        z.extractall(tmp_dir)
    for root, _, files in os.walk(tmp_dir):
        for name in files:
            fpath = os.path.join(root, name)
            try:
                res = load_file(fpath)
                out.extend(res.get("dataframes", []))
            except Exception:
                continue
    return out


def _read_tar(file_path: str) -> List[pd.DataFrame]:
    tmp_dir = tempfile.mkdtemp()
    out: List[pd.DataFrame] = []
    with tarfile.open(file_path, "r:*") as t:
        t.extractall(tmp_dir)
    for root, _, files in os.walk(tmp_dir):
        for name in files:
            fpath = os.path.join(root, name)
            try:
                res = load_file(fpath)
                out.extend(res.get("dataframes", []))
            except Exception:
                continue
    return out


# =========================
# PDF extractors (tables + text)
# =========================

def extract_pdf_tables(pdf_path: str, max_pages: Optional[str] = "all") -> List[pd.DataFrame]:
    """
    Robust multi-strategy PDF table extraction:
      1) Camelot lattice -> stream
      2) pdfplumber page.extract_tables (lines -> text strategies)
      3) Heuristic fixed-width fallback from text
    """
    dfs: List[pd.DataFrame] = []

    # 1) Camelot (great if ruled table & Ghostscript available)
    if camelot is not None:
        for flavor in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(pdf_path, flavor=flavor, pages=max_pages)
                for t in tables:
                    df = t.df
                    if df.shape[0] > 1:
                        df.columns = df.iloc[0].astype(str).str.strip()
                        df = df.iloc[1:].reset_index(drop=True)
                    df.columns = [str(c).strip() for c in df.columns]
                    if df.shape[1] > 1:
                        dfs.append(df)
                if dfs:
                    return dfs
            except Exception:
                pass  # continue to next strategy

    # 2) pdfplumber tables
    if pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    for settings in (
                        dict(vertical_strategy="lines", horizontal_strategy="lines"),
                        dict(vertical_strategy="text", horizontal_strategy="text"),
                    ):
                        try:
                            tables = page.extract_tables(settings)
                        except Exception:
                            tables = []
                        for rows in tables or []:
                            rows = [r for r in rows if any((cell or "").strip() for cell in r)]
                            if not rows or len(rows[0]) < 2:
                                continue
                            header = [str(c or "").strip() for c in rows[0]]
                            body = [[str(c or "").strip() for c in r] for r in rows[1:]]
                            df = pd.DataFrame(body, columns=header)
                            if df.shape[1] > 1:
                                dfs.append(df)
            if dfs:
                return dfs
        except Exception:
            pass

    # 3) Fallback: parse fixed-width text into a simple table
    text = extract_pdf_text(pdf_path, max_chars=100000)
    if text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        header_idx = -1
        header: List[str] = []
        for i, ln in enumerate(lines[:200]):
            parts = re.split(r"\s{2,}", ln)
            if len(parts) >= 2 and all(len(p) <= 40 for p in parts):
                header_idx = i
                header = [p.strip() for p in parts]
                break

        if header_idx >= 0 and header:
            body: List[List[str]] = []
            for ln in lines[header_idx + 1:]:
                parts = re.split(r"\s{2,}", ln)
                if len(parts) == len(header):
                    body.append([p.strip() for p in parts])
            if body:
                df = pd.DataFrame(body, columns=header)
                if df.shape[1] > 1:
                    dfs.append(df)

    return dfs


def extract_pdf_text(pdf_path: str, max_chars: int = 20000) -> str:
    """
    Extract raw text from a PDF via pdfplumber (best-effort).
    """
    if pdfplumber is None:
        return ""
    try:
        parts: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                if txt:
                    parts.append(txt)
                if sum(len(p) for p in parts) >= max_chars:
                    break
        return ("\n".join(parts))[:max_chars]
    except Exception:
       text=" "

    if len(text.strip()) < 200 :
        try:
            ocr_text = ocr_pdf_to_text(pdf_path)
            if ocr_text.strip():
                text = ocr_text[:max_chars]
        except Exception:
            pass

    return text

def ocr_pdf_to_text(pdf_path: str, max_pages: int = 10) -> str:
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import pytesseract
    except Exception:
        return ""

    doc = fitz.open(pdf_path)
    texts = []
    for i, page in enumerate(doc):
        if i >= max_pages: break
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        txt = pytesseract.image_to_string(img)
        if txt.strip():
            texts.append(txt)
    return "\n".join(texts)

# =========================
# Summaries
# =========================

def _summarize_dataframes(dfs: List[pd.DataFrame]) -> str:
    summaries: List[str] = []
    for i, df in enumerate(dfs):
        # sanitize column names to strings
        cols = [str(c) for c in df.columns]
        sample = df.head(3).to_dict(orient="records")
        summaries.append(
            f"DataFrame {i+1}\n"
            f"- shape: {df.shape}\n"
            f"- columns: {cols}\n"
            f"- sample (up to 3 rows): {sample}"
        )
    return "\n\n".join(summaries) if summaries else "No tabular data extracted."
