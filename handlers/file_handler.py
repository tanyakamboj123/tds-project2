
import os
import zipfile
import tarfile
import pandas as pd
import tempfile
from typing import List, Union,Optional
import pyarrow.parquet as pq
import duckdb
#from handlers.pdf_handler import load_pdf


# Optional imports for PDF
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import camelot  # pip install camelot-py[cv]
except Exception:
    camelot = None

def load_file(file_path: str) -> List[pd.DataFrame]:
    ext = os.path.splitext(file_path)[1].lower()
    dataframes = []

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, on_bad_lines='skip')  # Handles malformed CSV rows
            dataframes.append(df)

        elif ext == ".tsv":
            df = pd.read_csv(file_path, delimiter='\t', on_bad_lines='skip')
            dataframes.append(df)

        elif ext in [".xls", ".xlsx"]:
            xls = pd.ExcelFile(file_path)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                dataframes.append(df)
        
        elif ext == ".duckdb":
            con = duckdb.connect(file_path)
            table_names = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
            return [con.execute(f"SELECT * FROM {table}").fetchdf() for table in table_names]

        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
            dataframes.append(df)

        elif ext == ".zip":
            with zipfile.ZipFile(file_path, 'r') as archive:
                for name in archive.namelist():
                    with archive.open(name) as f:
                        tmp = tempfile.NamedTemporaryFile(delete=False)
                        tmp.write(f.read())
                        tmp.close()
                        dataframes += load_file(tmp.name)
                        os.unlink(tmp.name)

        elif ext == ".tar":
            with tarfile.open(file_path, 'r') as archive:
                for member in archive.getmembers():
                    if member.isfile():
                        f = archive.extractfile(member)
                        if f is not None:
                            tmp = tempfile.NamedTemporaryFile(delete=False)
                            tmp.write(f.read())
                            tmp.close()
                            dataframes += load_file(tmp.name)
                            os.unlink(tmp.name)

        
        elif ext == ".pdf":
            # Tables from PDF (best-effort)
            dataframes.extend(extract_pdf_tables(file_path))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        raise ValueError(f"Failed to load file {file_path}: {str(e)}")

    return dataframes

def summarize_dataframes(dfs: List[pd.DataFrame]) -> str:
    summary = []
    for i, df in enumerate(dfs):
        summary.append(f"DataFrame {i+1} - Shape: {df.shape}\nColumns: {list(df.columns)}\n")
    return "\n".join(summary)


# ------------- PDF helpers -------------

def extract_pdf_tables(pdf_path: str, max_pages: Optional[str] = "all") -> List[pd.DataFrame]:
    """
    Try to extract tables from PDF using Camelot.
    Returns list of DataFrames (may be empty if no tables or Camelot missing).
    """
    dfs: List[pd.DataFrame] = []
    if camelot is None:
        return dfs  # Camelot not installed → skip tables

    # First try lattice (works well on bordered tables), then stream
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(pdf_path, flavor=flavor, pages=max_pages)
            for t in tables:
                df = t.df
                # Normalize headers: promote first row to header if it looks like headers
                if df.shape[0] > 1:
                    df.columns = df.iloc[0].astype(str).str.strip()
                    df = df.iloc[1:].reset_index(drop=True)
                # Strip column names
                df.columns = [str(c).strip() for c in df.columns]
                dfs.append(df)
            if dfs:
                break  # if we got tables with this flavor, stop
        except Exception:
            continue
    return dfs


def extract_pdf_text(pdf_path: str, max_chars: int = 20000) -> str:
    """
    Extract text from a PDF using pdfplumber (best-effort).
    Returns a (possibly truncated) string. Empty string if pdfplumber not available.
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
                # short-circuit if we already have enough
                if sum(len(p) for p in parts) >= max_chars:
                    break
        text = "\n".join(parts)
        return text[:max_chars]
    except Exception:
        return ""


# ------------- Summaries for LLM prompt -------------

def summarize_dataframes(dfs: List[pd.DataFrame]) -> str:
    """
    Existing summary utility for DataFrames only.
    """
    summaries = []
    for i, df in enumerate(dfs):
        summaries.append(
            f"DataFrame {i+1} - shape: {df.shape}\n"
            f"Columns: {list(df.columns)}\n"
            f"Sample:\n{df.head(3).to_dict(orient='records')}\n"
        )
    return "\n".join(summaries) if summaries else "No tabular data extracted."


def summarize_dataframes_with_text(
    dfs: List[pd.DataFrame],
    text: str,
    text_label: str = "PDF TEXT (first ~2000 chars)"
) -> str:
    """
    Combined summary: tables (if any) + an excerpt of PDF text.
    Use this in your gemini_handler to build a richer prompt.
    """
    parts: List[str] = []
    parts.append(summarize_dataframes(dfs))
    if text:
        excerpt = text[:2000]
        parts.append(f"---- {text_label} ----\n{excerpt}")
    return "\n\n".join(p for p in parts if p.strip())