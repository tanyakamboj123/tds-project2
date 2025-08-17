# app.py
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
# ]
# ///

import io
import json
import base64
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from handlers.openai_handler import handle_question  # keep your existing handler

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

@app.get("/")
async def home():
    return {"message": "OK. POST your question and files to '/', or '/api/'."}

# Accept both / and /api/
@app.post("/")
@app.post("/api/")
async def analyze(
    request: Request,
    question: Optional[str] = Form(None),
    file: Optional[List[UploadFile]] = File(None),
):
    """
    Accepts:
    - multipart/form-data: question=<str> and 0..n files
      (or one .txt file containing the question + 0..n data files)
    - application/json: {"question": "...", "files":[{"name":"...", "content_b64":"..."}]}
    - text/plain: raw question body
    """
    try:
        q: Optional[str] = question
        uploaded: List[UploadFile] = file or []

        # Handle other content-types
        if q is None and request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            q = body.get("question")
            json_files = body.get("files", []) or []
            for jf in json_files:
                name = jf.get("name", "file.bin")
                content_b64 = jf.get("content_b64", "")
                data = base64.b64decode(content_b64) if content_b64 else b""
                uf = UploadFile(filename=name, file=io.BytesIO(data))
                uploaded.append(uf)

        if q is None and not uploaded:
            # Maybe raw text
            raw = await request.body()
            if raw:
                try:
                    # Allow raw to be either plain text question or JSON with "question"
                    as_json = json.loads(raw.decode("utf-8"))
                    q = as_json.get("question")
                except Exception:
                    q = raw.decode("utf-8")

        # If any uploaded file is .txt and we still don't have q, use its content as question
        if q is None:
            for uf in uploaded:
                if uf.filename.lower().endswith(".txt"):
                    q = (await uf.read()).decode("utf-8", errors="ignore")
                    await uf.seek(0)
                    break

        if not q:
            return JSONResponse({"error": "No question provided."}, status_code=400)

        # Persist uploaded files to tmp paths and pass to handler
        tmp_files = []
        for uf in uploaded:
            # Skip the question.txt if it was used for question (still OK to pass along)
            data = await uf.read()
            await uf.seek(0)
            fd, tmp_path = tempfile.mkstemp(suffix="-" + uf.filename)
            with open(tmp_path, "wb") as out:
                out.write(data)
            tmp_files.append({"name": uf.filename, "path": tmp_path})

        # Call your handler with the question AND the files list
        result = handle_question(q, file=tmp_files)  # <-- modify handler signature to accept list

        # Always return JSON
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# This part allows you to run the app with `python app.py` directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)