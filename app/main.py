# -----------------------
# CHAT (multi-turn)
# -----------------------
from fastapi import Request

SYSTEM_PROMPT = (
    "You are an investment research assistant. Be helpful, concise, and neutral. "
    "You can discuss crypto, stocks, ETFs, forex and macro news. "
    "Use disclaimers: you are not giving financial advice. "
    "If data is requested, you may call our API endpoints (e.g., /crypto/price/{id}, "
    "/stock/price/{ticker}, /dashboard) and summarize. If something is unknown, say so."
)

@app.post("/chat")
def chat(messages: Dict[str, Any] = Body(...)):
    """
    Body:
    {
      "messages": [
        {"role":"system/user/assistant", "content":"..."},
        ...
      ]
    }
    Returns: { "reply": "..." }
    """
    if not openai_client:
        raise HTTPException(400, "AI is not configured (set OPENAIKEY / OPENAI_API_KEY).")

    # Εξασφαλίζουμε ότι υπάρχει system prompt στην αρχή
    msgs = messages.get("messages") or []
    has_system = any(m.get("role") == "system" for m in msgs)
    if not has_system:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs

    try:
        # OpenAI Responses API (νέο SDK)
        resp = openai_client.responses.create(
            model="gpt-5-mini",
            input=[{"role": m["role"], "content": m["content"]} for m in msgs],
        )
        reply = resp.output_text.strip() if hasattr(resp, "output_text") else "No AI output."
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(502, f"AI error: {e}")

@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    # Απλό UI με ιστορικό στον browser (localStorage)
    return """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>AI Markets – Chat</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:900px;margin:32px auto;padding:0 16px}
h1{font-size:1.6rem;margin:0 0 8px}
#log{border:1px solid #eee;border-radius:8px;padding:12px;min-height:320px;white-space:pre-wrap}
.msg{margin:8px 0}
.user{color:#0b5;border-left:3px solid #0b5;padding-left:8px}
.assistant{color:#333;border-left:3px solid #999;padding-left:8px;background:#fafafa}
.row{display:flex;gap:8px;margin-top:12px}
input,button,textarea{font:inherit}
input{flex:1;padding:10px}
button{padding:10px 14px;cursor:pointer}
.small{color:#666;font-size:.9rem;margin-top:8px}
</style>
<h1>AI Markets – Chat</h1>
<p class="small">Ask anything in English (or Greek). This is educational research, not financial advice.</p>
<div id="log"></div>
<div class="row">
  <input id="q" placeholder="e.g. Where should I invest today considering high rates and BTC?">
  <button id="send">Send</button>
  <button id="clear" title="Clear conversation">Clear</button>
</div>
<p class="small">Tip: You can mention tickers/coins (AAPL, MSFT, SPY, bitcoin) and ask for analysis.</p>
<script>
const LOG = document.getElementById('log');
const Q = document.getElementById('q');
const BTN = document.getElementById('send');
const CLR = document.getElementById('clear');

function loadHistory(){
  try {
    return JSON.parse(localStorage.getItem('aimarkets_chat') || '[]');
  } catch(e){ return []; }
}
function saveHistory(h){ localStorage.setItem('aimarkets_chat', JSON.stringify(h)); }
function render(h){
  LOG.innerHTML = h.map(m => {
    const cls = m.role === 'user' ? 'user' : (m.role === 'assistant' ? 'assistant' : 'msg');
    return `<div class="msg ${cls}"><b>${m.role}:</b> ${escapeHtml(m.content)}</div>`;
  }).join('');
  LOG.scrollTop = LOG.scrollHeight;
}
function escapeHtml(s){ return s.replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c])); }

let history = loadHistory();
if(history.length === 0){
  history = [{role:"assistant", content:"Hi! I’m your research assistant. Ask me anything about markets."}];
  saveHistory(history);
}
render(history);

BTN.onclick = async () => {
  const text = Q.value.trim();
  if(!text) return;
  history.push({role:"user", content:text});
  render(history); saveHistory(history);
  Q.value = ""; Q.focus();
  try{
    const body = {messages: history};
    const r = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const j = await r.json();
    history.push({role:"assistant", content: j.reply || JSON.stringify(j)});
  }catch(e){
    history.push({role:"assistant", content: "Error: "+e});
  }
  render(history); saveHistory(history);
};
CLR.onclick = () => { history = []; saveHistory(history); location.reload(); };
</script>
</html>"""
