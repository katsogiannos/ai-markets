# app/main.py
import os
import logging
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------- Logging ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("ai-markets")

# ---------- App ----------
app = FastAPI(title="AI Markets Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ENV (strip to remove hidden whitespace/newlines) ----------
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
NEWS_API_KEY = (os.getenv("NEWS_API_KEY") or "").strip()
log.info("ENV CHECK | OPENAI_API_KEY present? %s", bool(OPENAI_API_KEY))
log.info("ENV CHECK | NEWS_API_KEY present? %s", bool(NEWS_API_KEY))

# ---------- OpenAI ----------
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-3.5-turbo"
HTTPX_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

async def call_openai_chat(query: str) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server is not configured with OpenAI API key.")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a cautious financial research assistant. You never give financial advice; only educational analysis."},
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            resp = await client.post(OPENAI_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError):
        raise HTTPException(status_code=502, detail="Connection error to OpenAI.")

    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid response from OpenAI.")

# ---------- Health ----------
@app.get("/")
async def root():
    return {"service": "ai-markets", "status": "ok"}

@app.get("/healthz")
async def healthz():
    return {"ok": True}

# ---------- Primary API ----------
@app.post("/advice")
async def advice(payload: Dict[str, Any]):
    query = (payload or {}).get("query") or (payload or {}).get("q") or ""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Missing 'query'.")
    text = await call_openai_chat(query)
    return {"result": text}

# ---------- Compatibility with your FRONTEND ----------
# Works if your page does GET /advisor?q=... or /advisor?query=...
@app.get("/advisor")
async def advisor_get(
    q: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    text: Optional[str] = Query(None),
    message: Optional[str] = Query(None),
    prompt: Optional[str] = Query(None),
):
    chosen = next((v for v in [q, query, text, message, prompt] if v), "")
    if not chosen.strip():
        # keep same shape so your frontend prints something meaningful
        return {"result": "Missing query. Use ?q=... or POST /advice with {\"query\":\"...\"}."}
    res = await call_openai_chat(chosen)
    return {"result": res}

# Also accept POST /advisor with same body as /advice
@app.post("/advisor")
async def advisor_post(payload: Dict[str, Any]):
    query = (payload or {}).get("query") or (payload or {}).get("q") or ""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Missing 'query'.")
    res = await call_openai_chat(query)
    return {"result": res}
