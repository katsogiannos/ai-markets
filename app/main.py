import os
import json
import logging
from typing import Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

# -----------------------
# Logging (για debug στο Render)
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("aimarkets")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="AI Markets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Environment variables
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # προετοιμασμένο για /news όταν το ανοίξουμε

log.info("ENV CHECK | OPENAI_API_KEY present? %s", bool(OPENAI_API_KEY))
log.info("ENV CHECK | NEWS_API_KEY present? %s", bool(NEWS_API_KEY))

# -----------------------
# Helpers
# -----------------------
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-3.5-turbo"  # σταθερό και διαθέσιμο

async def call_openai_chat(query: str) -> str:
    """
    Κλήση στο OpenAI Chat Completions API με httpx.
    Επιστρέφει μόνο το text (assistant message) ή πετάει HTTPException.
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
            {"role": "system", "content": "You are a helpful, cautious financial research assistant. You never give financial advice; only educational analysis."},
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    log.info("OPENAI CALL | sending request…")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OPENAI_URL, headers=headers, json=payload)
    except httpx.RequestError as e:
        log.exception("OPENAI CALL | network error: %s", str(e))
        raise HTTPException(status_code=502, detail="Connection error to OpenAI.")

    if resp.status_code >= 400:
        # γράφουμε το body στα logs για troubleshooting
        try:
            err_body = resp.json()
        except Exception:
            err_body = {"raw": resp.text}
        log.error("OPENAI CALL | HTTP %s | %s", resp.status_code, err_body)
        # Μην επιστρέφεις το κλειδί/headers – μόνο γενικό σφάλμα στον χρήστη
        raise HTTPException(status_code=502, detail="OpenAI returned an error.")

    data = resp.json()
    log.info("OPENAI CALL | ok.")
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        log.error("OPENAI CALL | unexpected response shape: %s", json.dumps(data)[:800])
        raise HTTPException(status_code=502, detail="Unexpected response from OpenAI.")

# -----------------------
# Routes
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h2>AI Markets is running.</h2>
    <p>Try: <a href="/healthz">/healthz</a> · <a href="/advisor">/advisor</a></p>
    """

@app.get("/healthz")
async def healthz():
    return {"ok": True}

# --- Advisor UI (free text) ---
# Απλή σελίδα HTML (χωρίς templates) που καλεί το /api/advice
@app.get("/advisor", response_class=HTMLResponse)
async def advisor_page():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AI Markets Advisor</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 28px; max-width: 980px; margin: 0 auto; }
    h1 { margin: 0 0 12px 0; }
    textarea { width: 100%; height: 220px; font-family: inherit; font-size: 15px; padding: 10px; }
    button { padding: 10px 18px; font-size: 14px; cursor: pointer; }
    .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 18px; margin-top: 18px; background: #fafafa; }
    pre { background: #0f172a; color: #e2e8f0; padding: 10px; overflow: auto; border-radius: 8px; }
    .muted { color: #6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <h1>AI Markets Advisor</h1>
  <p class="muted">Type anything you want (free text). This is educational, not financial advice.</p>
  <textarea id="q" placeholder="Where should I invest today?"></textarea>
  <br/><br/>
  <button onclick="run()">Analyze</button>

  <div class="card">
    <div class="muted">Result</div>
    <pre id="out"></pre>
  </div>

<script>
async function run() {
  const out = document.getElementById('out');
  const q = document.getElementById('q').value.trim();
  out.textContent = 'Working...';
  try {
    const r = await fetch('/api/advice', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ query: q })
    });
    const data = await r.json();
    if (r.ok) {
      out.textContent = data.result || '';
    } else {
      out.textContent = 'Error: ' + (data.detail || data.error || 'unknown error');
    }
  } catch (e) {
    out.textContent = 'Error: ' + e.toString();
  }
}
</script>
</body>
</html>
    """

# --- Advisor API ---
@app.post("/api/advice")
async def api_advice(payload: Dict[str, Any]):
    query = (payload or {}).get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query.")
    log.info("ADVICE | query: %s", query[:200])
    text = await call_openai_chat(query)
    # Μικρή καθαριότητα του κειμένου
    return JSONResponse({"result": text.strip()})

# -----------------------
# Error handlers (πιο καθαρά μηνύματα)
# -----------------------
@app.exception_handler(HTTPException)
async def http_exc_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exc_handler(_, exc: Exception):
    log.exception("UNHANDLED | %s", str(exc))
    return JSONResponse(status_code=500, content={"detail": "Server error."})

