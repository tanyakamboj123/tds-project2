# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "google-genai",
# ]
# ///
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from handlers.openai_handler import handle_question
import uvicorn

app = FastAPI()

# CORS (open for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}

@app.post("/api/")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    question = contents.decode("utf-8")
    return handle_question(question)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
