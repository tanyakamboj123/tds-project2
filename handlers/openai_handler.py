# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "google-genai",
#   "python-dotenv",
#   "numpy",
#   "matplotlib",
#   "pandas",
#   "requests",
#   "scikit-learn",
#   "beautifulsoup4",
#   "lxml",
#   "scipy",
#   "io",
# ]
# ///
import os
import io
import json
import traceback
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
from handlers.url_scraper import extract_url, scrape_url
from handlers.file_handler2 import load_file, summarize_context
#from handlers.file_handler import load_file, extract_pdf_text, summarize_dataframes_with_text
#from handlers.pdf_handler import load_pdf
import tempfile
from fastapi import UploadFile,Request
from typing import List, Optional 
import base64
from io import BytesIO
from PIL import Image  # pip install pillow
  


load_dotenv()
api_key = os.getenv("AIPIPE_TOKEN")
base_url = "https://aipipe.org/openai/v1"

MODEL_NAME = "gpt-4o-mini"

def extract_url(text):
    match = re.search(r'(https?://[^\s]+)', text)
    return match.group(1) if match else None

def scrape_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Get all visible text
        text = soup.get_text(separator='\n', strip=True)

        # Optionally: extract tables, titles, etc. later
        return text[:15000]  # Truncate to avoid token limit

    except Exception as e:
        return f"Error fetching URL: {e}"
    
    
def fig_to_base64_png(fig, max_size_kb=100, dpi_primary=80, dpi_fallback=60):
    """Save a Matplotlib figure to a base64 PNG under `max_size_kb`."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi_primary, bbox_inches="tight")
    data = buf.getvalue()

    if len(data) > max_size_kb * 1024:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi_fallback, bbox_inches="tight")
        data = buf.getvalue()

    return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")


async def handle_question(request: Request):
    """
    Unified handler:
    - Parses multipart form
    - Gets 'question' or infers from a .txt file
    - Processes any uploaded files (all of them)
    - Scrapes URL context if a URL is present in the question
    - Calls the model, extracts Python code, executes it, returns 'result'
    """
    try:
        # 1) Parse form
        form = await request.form()
        question: Optional[str] = form.get("question")
        files: List[UploadFile] = form.getlist("file") if "file" in form else []

        # 2) If no explicit question, try to infer from a .txt file
        if not question:
            for f in files:
                if hasattr(f, "filename") and f.filename and f.filename.lower().endswith(".txt"):
                    content = await f.read()
                    await f.seek(0)
                    question = content.decode("utf-8", errors="ignore").strip()
                    break

        if not question:
            return {"error": "No question provided."}

        # 3) URL context (if present in question)
        url = extract_url(question)
        url_context = scrape_url(url) if url else ""

        # 4) File contexts (process ALL files)
        tmp_paths: List[str] = []
        file_context_chunks: List[str] = []

        try:
            for f in files:
                # Write each upload to a temp file
                data = await f.read()
                await f.seek(0)

                fd, tmp_path = tempfile.mkstemp(suffix="-" + (f.filename or "upload"))
                try:
                    with os.fdopen(fd, "wb") as out:
                        out.write(data)
                except Exception:
                    # If os.fdopen fails, make sure the fd is closed
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                    raise
                tmp_paths.append(tmp_path)

                # Use your generic loader + summarizer
                try:
                    load_result = load_file(tmp_path)          # list of dfs, text, image OCR, etc. as your file_handler2 supports
                    summary = summarize_context(load_result)   # turn into compact text for LLM
                    if summary:
                        file_context_chunks.append(summary)
                except Exception as fe:
                    file_context_chunks.append(f"[File {f.filename}] Error: {fe}")

            file_context = "\n\n".join(file_context_chunks).strip()
        finally:
            # 5) Cleanup temp files
            for p in tmp_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

        # 6) Build prompt
        prompt = (
        "You are an expert and precise python data analyst.\n\n"
        "Task:\n"
        "- The user will give you a question can be related to url and any type of file.\n"
        "- Break the task into smaller sub-questions.\n"
        "- Ignore the irrelevant data while extraction.\n"
        "- If a URL is provided, extract data using requests + BeautifulSoup or pandas.read_html() or any other libraries.\n"
        #"- Extract the data which is relevant to the question being asked and ignore the unimportant details.\n"
        "- If a file is provided either it is local extract the data from the file.\n"
        "- Never respond with text; always write valid Python code that processes the provided context.\n"
        "- For PDFs, use the libraries suitable for the question being asked.\n"
        "    - use the most appropriate libraries for PDF data extraction like pdfplumber and others.\n"
        "    - If asked for column names, answer **only** from the extracted DataFrame columns shown in CONTEXT (do not guess).\n"
        "    - Perform the operations on the column data of table if asked in the question after extracting the relevant data.\n"
        "- Use the libraries used in the connecting python files.\n"
        "- Use the provided context to answer the question.\n"
        "- Answer all sub-questions using code.\n"
        "- while dealing with files consider the statement of the question regardless of the upper or lower case of the column names.\n"
        "- Your final output MUST be stored in a variable called `result`.\n"
        "-For parquet files:"
            "When you need counts of rows in parquet files:" 
              "use `parquet_metadata(...)` instead of `read_parquet(...)`."
              "Do not load PDFs, TARs, or JSONs unless explicitly asked."
              " Always prefer metadata-only queries for large S3 parquet datasets."
           
        "- For images:\n"
        "    - Use the suitable libraries to deal with the image questions.\n"
        "    - Load the image and understand the question carefully and analyse the image accordingly. And give the answer.\n"
        "    - use easyocr to extract the text and do not use pytesseract.\n\n"
       
        "Output format:\n"
        "- result must be a Python **list of strings or JSON serializable**\n"
        "- Only answers to be printed with no description.\n"
        "- The **last line** of your code must be:\n"
        "    print(json.dumps(result))\n\n"
        "- Do not make arrays of different answers you got from different questions asked."

        "If a plot is requested , make this chart if asked only:\n"
        "- Use matplotlib\n"
        "- If you create any Matplotlib figure, always convert it with fig_to_base64_png(fig) so the base64 PNG is < 100 kB."
        #"- Save to base64 string as \"data:image/png;base64,...\" and include it in the result array\n"
        "- When generating plots, compress and resize the figure so that the base64-encoded output is under 100 KB. Use PIL (Pillow) to downscale and optimize images. Always return only the base64 string (no prefix)\n"
        "- Read the question carefully what is asked then generate the plot.\n\n"
        
        f"{('[URL]\n' + url_context) if url_context else ''}"
        f"{('[FILES]\n' + file_context) if file_context else ''}"
        
        "Before numeric operations like .corr(), .plot(), or .astype(), always:\n"
        "- Convert columns using pd.to_numeric(..., errors='coerce'), OR\n"
        "- Use .str.extract(r'(\\d+(?:\\.\\d+)?)') to extract numeric parts from strings like '24RK' or 'T2257844554' if in present in another format.\n"
        "- Drop missing (NaN) values with .dropna() before computing\n"
        "- convert strings to numeric using pd.to_numeric(..., errors='coerce').\n"
        "- When handling mixed data (numbers + text), use pd.to_numeric(errors=\"coerce\") instead of astype(float).\n\n"
        "- Use try/except to guard risky code like .iloc, .astype, .droplevel.\n"
        "- Before calling .astype(float), always verify if column is numeric or use pd.to_numeric(..., errors='coerce').\n\n"
        "- Ignore special characters like $,# etc in numeric columns before conversion.\n"
        
        "- Reduce variability in the answers, use consistent formatting and rounding.\n"
        "- If any error gets , send back that error to llm to solve the question again.\n\n"
        "- take maximum of 3 minutes to answer the question.\n\n"
        "Strictly read all the instructions mentioned above before solving the question."
        " Now generate Python code to answer:"
          f"{question}"
    )
        # Call gpt-4o-mini via AIpipe
        # ✅ Call AIpipe REST API instead of OpenAI client
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a Python data analyst who specializes in data extraction and analysis."},
                    {"role": "user", "content": prompt},
                    ],
               # response_format={"type":"json_object"}
                "temperature": 0.2
            },
            timeout=120
        )

        if response.status_code != 200:
            return {"error": f"API request failed: {response.status_code}", "details": response.text}

        reply = response.json()["choices"][0]["message"]["content"]
        
        def extract_code(text: str) -> str:
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
            return code_blocks[0] if code_blocks else text
        
        code = extract_code(reply)
         # Optional: print for debugging
        #print("\n======= PROMPT SENT TO GEMINI =======\n")
        #print(prompt[:800] + "...\n")
        print("======= GENERATED PYTHON CODE =======\n")
        print(code)


       
        exec_locals = {}

        exec(code,  exec_locals)

        result = exec_locals.get("result", {"status": "result not set"})
        return result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }