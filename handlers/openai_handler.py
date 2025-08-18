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
from fastapi import UploadFile
from typing import List, Optional   


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

def handle_question(question: str, file: Optional[List[dict]] = None):
    try:
        url = extract_url(question)
        context = ""

        if url:
            context = scrape_url(url)
        elif file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.file.read())
                tmp_path = tmp.name

            load_result = load_file(tmp_path)
            context = summarize_context(load_result)
        prompt = prompt = (
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
        #"- Save to base64 string as \"data:image/png;base64,...\" and include it in the result array\n"
        "- Make sure base64 output is under 10,000 characters\n"
        "- Read the question carefully what is asked then generate the plot.\n\n"
        f"Data from webpage:\n{context if context else ''}\n\n"
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
        "Strictly read all the instructions mentioned above before solving the question.\n\n"
        "Now generate Python code to answer:\n\n"
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