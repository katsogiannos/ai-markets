# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI(title="AI Markets")

# ---------------------------
# Helpers
# ---------------------------
def _get_openai_key() -> str:
    # Δοκιμάζουμε τα πιο συνηθισμένα ονόματα ENV για το κλειδί
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPEN_AI_KEY")
        or os.getenv("OPENAIKEY")
        or ""
    )

def _make_client() -> OpenAI:
    key = _get_openai_key()
    if not key:
        raise RuntimeError(
            "Missing OpenAI API key (set OPENAI_API_KEY ή OPEN_AI_KEY στο Render → Environment)."
        )
    return OpenAI(api_key=key)

# ---------------------------
# DTOs
# ---------------------------
class AskPayload(BaseModel):
    question: str

# ---------------------------
# Routes
# ---------------------------
@app.get("/healthz", response_class=JSONResponse)
def healthz():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home():
    return """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>AI Markets</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:40px;max-width:900px}
    textarea,input,button{font:inherit}
    textarea{width:100%;height:180px;padding:10px}
    .row{display:grid;gap:10px}
    .card{border:1px solid #ddd;border-radius:8px;padding:16px}
    pre{white-space:pre-wrap;word-break:break-word;background:#0d1117;color:#e6edf3;padding:12px;border-radius:8px}
    .muted{color:#666}
  </style>
</head>
<body>
  <h1>AI Markets is running.</h1>
  <p class="muted">Try: <a href="/healthz">/healthz</a> · <a href="/advisor">/advisor</a></p>
</body>
</html>
"""

@app.get("/advisor", response_class=HTMLResponse)
def advisor_page():
    # Απλή σελίδα όπου ο χρήστης γράφει ελεύθερα την ερώτηση
    return """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>AI Markets Advisor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:40px;max-width:920px}
    textarea,button{font:inherit}
    textarea{width:100%;height:200px;padding:10px}
    button{padding:10px 16px;border:1px solid #444;border-radius:6px;background:#111;color:#fff;cursor:pointer}
    .card{border:1px solid #ddd;border-radius:8px;padding:16px;margin-top:20px}
    pre{white-space:pre-wrap;word-break:break-word;background:#0d1117;color:#e6edf3;padding:12px;border-radius:8px}
  </style>
</head>
<body>
  <h1>AI Markets Advisor</h1>
  <p>Type anything you want (free text). This is educational, not financial advice.</p>

  <div class="card">
    <textarea id="q" placeholder="WHERE SHOULD I INVEST TODAY?"></textarea>
    <div style="margin-top:10px">
      <button id="btn">Analyze</button>
    </div>
    <div class="card" id="out" style="display:none">
      <h3>Result</h3>
      <pre id="res"></pre>
    </div>
  </div>

<script>
const btn = document.getElementById('btn');
const q = document.getElementById('q');
const out = document.getElementById('out');
const res = document.getElementById('res');

btn.onclick = async () => {
  res.textContent = 'Working...';
  out.style.display = 'block';
  try {
    const r = await fetch('/api/advisor', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: q.value || ''})
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || JSON.stringify(data));
    res.textContent = data.answer || JSON.stringify(data, null, 2);
  } catch (err) {
    res.textContent = 'Error: ' + err.message;
  }
};
</script>
</body>
</html>
"""

@app.post("/api/advisor", response_class=JSONResponse)
def advisor_api(payload: AskPayload):
    """
    Δέχεται ελεύθερο κείμενο από τον χρήστη και ζητά περίληψη/πρόταση από το OpenAI.
    (Αυστηρά για educational use — όχι επενδυτική συμβουλή.)
    """
    try:
        question = (payload.question or "").strip()
        if not question:
            return JSONResponse({"detail": "Please provide a question."}, status_code=400)

        client = _make_client()

        system = (
            "You are an investment research assistant. "
            "Provide careful, cautious, educational analysis. "
            "Avoid giving financial advice; emphasize risks and disclaimers."
        )
        user = f"User question:\n{question}\n\nReturn a concise, structured answer in English."

        # OpenAI Responses API (SDK v1+)
        resp = client.responses.create(
            model="gpt-5-mini",
            input=[{"role":"system","content":system},{"role":"user","content":user}],
            # Optional: small guardrails to keep it short and crisp
            max_output_tokens=600,
        )
        answer = getattr(resp, "output_text", None) or "No answer."

        return {"answer": answer}

    except Exception as e:
        # Επιστρέφουμε το error για debugging στο UI
        return JSONResponse({"detail": f"AI error: {e}"}, status_code=500)
