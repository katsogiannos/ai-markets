# app/main.py
import os
import logging
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("ai-markets")

# -------------------------------------------------
# FastAPI
# -------------------------------------------------
app = FastAPI(title="AI Markets Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Environment variables
# -------------------------------------------------
# .strip() FIX: αφαιρεί κενά/newlines που έσπαγαν το Authorization header
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
NEWS_API_KEY = (os.getenv("NEWS_API_KEY") or "").strip()

log.info("ENV CHECK | OPENAI_API_KEY present? %s", bool(OPENAI_API_KEY))
log.info("ENV CHECK | NEWS_API_KEY present? %s", bool(NEWS_API_KEY))

# -------------------------------------------------
# Constants
# -------------------------------------------------
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-3.5-turbo"  # σταθερό/διαθέσιμο

HTTPX_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
async def call_openai_chat(query: str) -> str:
    """
    Κλήση στο OpenAI Chat Completions API με httpx.
    Επιστρέφει μόνο το assistant text ή πετάει HTTPException.
    """
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY is missing in environment.")
        raise HTTPException(status_code=500, detail="Server is not configured with OpenAI API key.")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are a helpful, cautious financial research assistant. "
                "You never give financial advice; only educational analysis."
            )},
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    log.info("OPENAI CALL | sending request...")
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            resp = await client.post(OPENAI_URL, headers=headers, json=payload)
            txt_preview = (resp.text or "")[:300].replace("\n", " ")
            log.info("OPENAI RESPONSE | status=%s, body[0:300]=%s", resp.status_code, txt_preview)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        # 4xx/5xx από OpenAI
        status = e.response.status_code if e.response is not None else 502
        body = (e.response.text if e.response is not None else "")[:300]
        log.error("OPENAI HTTPStatusError | status=%s body=%s", status, body)
        if status == 401:
            raise HTTPException(status_code=502, detail="Connection error to OpenAI.")  # frontend-friendly
        raise HTTPException(status_code=502, detail="Connection error to OpenAI.")
    except httpx.RequestError as e:
        # network/DNS/SSL/timeouts
        log.error("OPENAI RequestError | %s", str(e))
        raise HTTPException(status_code=502, detail="Connection error to OpenAI.")
    except Exception as e:
        log.exception("OPENAI CALL FAILED | unexpected")
        raise HTTPException(status_code=502, detail="Connection error to OpenAI.")

    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        log.error("OPENAI PARSE ERROR | data=%s", str(data)[:400])
        raise HTTPException(status_code=502, detail="Invalid response from OpenAI.")

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/advice")
async def advice(payload: Dict[str, Any]):
    """
    Body: { "query": "text" }
    """
    query = (payload or {}).get("query") or ""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Missing 'query'.")

    log.info("ADVICE | query_len=%d", len(query))
    text = await call_openai_chat(query)
    return {"result": text}

# -------------------------------------------------
# Root (optional)
# -------------------------------------------------
@app.get("/")
async def root():
    return {"service": "ai-markets", "status": "ok"}


