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
import google.generativeai as genai
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
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

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

def handle_question(question: str, file: UploadFile = None):
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
        prompt = f"""
You are an expert and precise python data analyst.

Task:
- The user will give you a question can be related to url and any type of file.
- Break the task into smaller sub-questions and extract the metadata.
- Ignore the irrelevant data while extraction.
- If a URL is provided, extract data using requests + BeautifulSoup or pandas.read_html() or any other libraries.
- Extract the data which is relevant to the question being asked and ignore the unimportant details.
- If a file is provided either it is local extract the data from the file.
- Never respond with text; always write valid Python code that processes the provided context.
- For PDFs, use the libraries suitable for the question being asked.
    - use the most appropriate libraries for PDF data extraction like pdfplumber and others.
    - If asked for column names, answer **only** from the extracted DataFrame columns shown in CONTEXT (do not guess).
    - Perform the operations on the column data of table if asked in the question after extracting the relevant data.
- Use the libraries used in the connecting python files.
- Use the provided context to answer the question.
- Answer all sub-questions using code.
-while dealing with files consider the statement of the question regardless of the upper or lower case of the column names.
- Your final output MUST be stored in a variable called `result`.
- For images:
    - Use the suitable libraries to deal with the image questions.
    - Load the image and understand the question carefully and analyse the image accordingly. And give the answer.
    - use easyocr to extract the text and do not use pytesseract.

Output format:
- result must be a Python **list of strings or JSON serializable**
- Only answers to be printed with no description.
- The **last line** of your code must be:
    print(json.dumps(result))

If a plot is requested , make this chart if asked only:
- Use matplotlib
- Save to base64 string as "data:image/png;base64,..." and include it in the result array
- Make sure base64 output is under 10,000 characters
- Read the question carefully what is asked then generate the plot.

Data from webpage:
{"Here is a webpage that may be useful:\n" + context if context else ""}

Before numeric operations like .corr(), .plot(), or .astype(), always:
- Convert columns using pd.to_numeric(..., errors='coerce'), OR
- Use .str.extract(r'(\\d+(?:\\.\\d+)?)') to extract numeric parts from strings like '24RK'
- Drop missing (NaN) values with .dropna() before computing
- convert strings to numeric using pd.to_numeric(..., errors='coerce').
- When handling mixed data (numbers + text), use pd.to_numeric(errors="coerce") instead of astype(float).

- Use try/except to guard risky code like .iloc, .astype, .droplevel.
- Before calling .astype(float), always verify if column is numeric or use pd.to_numeric(..., errors='coerce').

- If any error gets , send back that error to llm to solve the question again.

-take maximum 3 minutes to answer the question.

Strictly follow all the instuctions mentioned above.

Now generate Python code to answer:

{question}
"""

        # Call gpt-4o-mini via AIpipe
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a Python data analyst who specializes in data extraction and analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        reply = response.choices[0].message.content
        #code = response.text.strip("```python").strip("```")
        
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