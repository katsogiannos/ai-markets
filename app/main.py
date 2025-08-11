import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="AI Markets")

# CORS (για να παίζει ο web client)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Διαβάζουμε key από διάφορες πιθανές μεταβλητές περιβάλλοντος
OPENAI_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPEN_AI_KEY")
    or os.getenv("OPENAIKEY")
    or ""
)

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>AI Markets is running.</h2>
    <p>Try: <a href="/healthz">/healthz</a>, <a href="/advisor">/advisor</a></p>
    """

# Πολύ απλό UI για ελεύθερο prompt
@app.get("/advisor", response_class=HTMLResponse)
def advisor_page():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AI Markets Advisor</title>
  <style>
    body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 16px}
    textarea,input,button{font:inherit}
    textarea{width:100%;min-height:120px;padding:10px}
    .row{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center}
    button{padding:10px 16px}
    pre{background:#111;color:#eee;padding:14px;border-radius:6px;white-space:pre-wrap}
  </style>
</head>
<body>
  <h1>AI Markets Advisor</h1>
  <p>Type anything you want (free text). This is educational, not financial advice.</p>
  <textarea id="q" placeholder="Where should I invest today?"></textarea>
  <div class="row">
    <small id="status"></small>
    <button id="btn">Analyze</button>
  </div>
  <h3>Result</h3>
  <pre id="out"></pre>

<script>
  const btn = document.getElementById('btn');
  const q = document.getElementById('q');
  const out = document.getElementById('out');
  const status = document.getElementById('status');

  btn.onclick = async () => {
    out.textContent = '';
    status.textContent = 'Working...';
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({message: q.value})
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      out.textContent = data.answer;
    } catch (e) {
      out.textContent = 'Error: ' + e.message;
    } finally {
      status.textContent = '';
    }
  };
</script>
</body>
</html>
    """

@app.post("/chat")
async def chat(req: Request):
    """
    Δέχεται: {"message": "..."} και επιστρέφει {"answer": "..."}.
    Αν υπάρχει OpenAI key, καλεί OpenAI. Αλλιώς, κάνει safe fallback.
    """
    body = await req.json()
    user_msg = (body or {}).get("message", "").strip()

    if not user_msg:
        return JSONResponse({"detail": "Empty message."}, status_code=400)

    # Αν δεν υπάρχει key, απαντάμε απλά για να μην σκάει.
    if not OPENAI_KEY:
        demo = (
            "⚠️ No OpenAI key configured on server.\n\n"
            "Echo of your request:\n---\n"
            f"{user_msg}\n---\n\n"
            "Add OPENAI_API_KEY (or OPEN_AI_KEY / OPENAIKEY) in Render → Environment."
        )
        return {"answer": demo}

    # Κλήση OpenAI Responses API (stream=false)
    try:
        payload = {
            "model": "gpt-5-mini",
            "input": f"You are a cautious markets research assistant. "
                     f"Answer concisely and clearly. The user asked:\n{user_msg}"
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://api.openai.com/v1/responses", json=payload, headers=headers)
            if r.status_code != 200:
                return JSONResponse({"detail": f"OpenAI error: {r.text}"}, status_code=502)
            data = r.json()
            # Προσπαθούμε να διαβάσουμε ένα απλό text απόκρισης
            # (το schema των responses μπορεί να έχει 'output_text' ή 'choices' ανάλογα με το μοντέλο).
            text = (
                data.get("output_text")
                or (data.get("choices") or [{}])[0].get("message", {}).get("content")
                or str(data)
            )
            return {"answer": text}
    except Exception as e:
        return JSONResponse({"detail": f"AI error: {e}"}, status_code=500)


