# app.py
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
# ]
# ///

import tempfile
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from handlers.openai_handler import handle_question  # keep your existing handler

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

@app.get("/")
async def home():
    return {"message": "OK. POST your question and files to '/api/'."}


# app.py (POST handler)
@app.post("/")
@app.post("/api/")
async def analyze(request: Request):
    try:
        result = await handle_question(request)  # <-- await the async handler
        return JSONResponse({"result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
