from fastapi import FastAPI
from fastapi.responses import HTMLResponse
app = FastAPI()

@app.get("/healthz")
def health(): 
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home():
    return "It works ✅"
