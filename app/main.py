# app/main.py
import os, logging
from typing import Any, Dict, Optional
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("ai-markets")

app = FastAPI(title="AI Markets Advisor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-3.5-turbo"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

async def call_openai(query: str) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(500, "Server is not configured with OpenAI API key.")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a cautious financial research assistant. No financial advice."},
            {"role": "user", "content": query},
        ],
        "temperature": 0.2, "max_tokens": 800
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as cl:
            r = await cl.post(OPENAI_URL, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except (httpx.RequestError, httpx.HTTPStatusError):
        raise HTTPException(502, "Connection error to OpenAI.")
    except Exception:
        raise HTTPException(502, "Invalid response from OpenAI.")

@app.get("/")
async def root(): return {"service":"ai-markets","status":"ok"}
@app.get("/healthz")
async def healthz(): return {"ok": True}

# --- Single handler ---
async def _handle_advice(payload: Dict[str, Any]) -> Dict[str, Any]:
    q = (payload or {}).get("query") or (payload or {}).get("q") or ""
    if not q.strip():
        raise HTTPException(400, "Missing 'query'.")
    return {"result": await call_openai(q)}

# --- Routes your frontend may call ---
@app.post("/advice")
async def advice(payload: Dict[str, Any]): return await _handle_advice(payload)

@app.post("/api/advice")                   # <-- ΝΕΟ για το /api/advice
async def api_advice(payload: Dict[str, Any]): return await _handle_advice(payload)

@app.get("/advisor")                       # GET /advisor?q=...
async def advisor_get(
    q: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    text: Optional[str] = Query(None),
    message: Optional[str] = Query(None),
    prompt: Optional[str] = Query(None),
):
    chosen = next((v for v in [q, query, text, message, prompt] if v), "")
    if not chosen.strip():
        return {"result": "Missing query. Use ?q=... or POST /advice with {\"query\":\"...\"}."}
    return {"result": await call_openai(chosen)}

@app.post("/advisor")                      # POST /advisor {query:"..."}
async def advisor_post(payload: Dict[str, Any]): return await _handle_advice(payload)
@app.get("/debug-openai")
async def debug_openai():
    import httpx, json, os
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
        return {"status": r.status_code, "body": r.text[:500]}
    except Exception as e:
        return {"status": "request_failed", "error": str(e)}
