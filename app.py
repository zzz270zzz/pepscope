"""
PepScope - Peptide Multi-Function Classifier Web App
FastAPI + embedded HTML. Run with: uvicorn app:app --host 0.0.0.0 --port 6006
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from src.inference import get_predictor

app = FastAPI(title="PepScope", version="1.0.0")


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PepScope - Peptide Classifier</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: #f0f2f5;
         color: #333; min-height: 100vh; }
  .container { max-width: 860px; margin: 0 auto; padding: 20px; }
  header { background: linear-gradient(135deg, #0d1b4a, #1a3a7a); color: #fff;
           padding: 28px 24px; border-radius: 14px; margin-bottom: 20px; }
  header h1 { font-size: 22px; font-weight: 600; letter-spacing: 0.5px; }
  header h1 span { opacity: 0.7; font-weight: 300; }
  header p { opacity: 0.75; font-size: 13px; margin-top: 4px; }
  .card { background: #fff; border-radius: 12px; padding: 20px 24px;
          box-shadow: 0 1px 8px rgba(0,0,0,0.07); margin-bottom: 16px; }
  textarea { width: 100%; min-height: 110px; padding: 12px; border: 2px solid #e0e0e0;
             border-radius: 8px; font-family: 'SF Mono', 'Courier New', monospace;
             font-size: 14px; resize: vertical; transition: border 0.2s; }
  textarea:focus { outline: none; border-color: #1a3a7a; }
  .hint { font-size: 12px; color: #888; margin-top: 6px; }
  .examples { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
  .example-btn { background: #eef1ff; color: #1a3a7a; border: 1px solid #ccd4f0;
                 padding: 3px 12px; border-radius: 14px; font-size: 12px;
                 cursor: pointer; transition: all 0.2s; }
  .example-btn:hover { background: #dde3ff; border-color: #1a3a7a; }
  .row { display: flex; align-items: center; gap: 12px; margin-top: 12px; }
  button.primary { background: #1a3a7a; color: #fff; border: none; padding: 10px 28px;
                   border-radius: 8px; font-size: 15px; font-weight: 500;
                   cursor: pointer; transition: background 0.2s; }
  button.primary:hover { background: #0d1b4a; }
  button.primary:disabled { background: #8899cc; cursor: wait; }
  #spinner { display: none; color: #666; font-size: 14px; }
  #spinner.active { display: inline; }

  .res-card { border: 1px solid #e8e8e8; border-radius: 10px; padding: 16px 20px;
              margin-bottom: 14px; }
  .res-seq { font-family: 'SF Mono', monospace; font-size: 13px; color: #444;
             word-break: break-all; background: #f8f9fc; padding: 6px 10px;
             border-radius: 6px; margin-bottom: 12px; }
  .summary { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
  .tag { display: inline-flex; align-items: center; gap: 4px; padding: 4px 12px;
         border-radius: 20px; font-size: 13px; font-weight: 500; }
  .tag-strong { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
  .tag-medium { background: #fff8e1; color: #e65100; border: 1px solid #ffcc80; }
  .tag-weak   { background: #fce4ec; color: #880e4f; border: 1px solid #f48fb1; }
  .tag-none   { background: #f5f5f5; color: #999; border: 1px solid #ddd; }

  .bar-row { display: flex; align-items: center; margin: 3px 0; gap: 8px; }
  .bar-label { width: 100px; font-size: 12px; text-align: right; flex-shrink: 0;
               font-weight: 500; color: #555; }
  .bar-track { flex: 1; height: 20px; background: #edf0f7; border-radius: 4px;
               overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease;
              display: flex; align-items: center; justify-content: flex-end;
              padding-right: 5px; font-size: 10px; color: #fff; font-weight: 700;
              min-width: 28px; }
  .bar-pct { width: 48px; font-size: 12px; color: #555; font-family: monospace;
             text-align: right; }
  .top-hit { margin-top: 10px; padding: 8px 12px; background: #f8f9fc;
             border-radius: 6px; font-size: 13px; }
  .top-hit strong { color: #1a3a7a; }
  .error-box { background: #ffebee; color: #b71c1c; padding: 10px 14px;
               border-radius: 8px; font-size: 14px; }
  footer { text-align: center; padding: 24px; font-size: 12px; color: #999; }
  @media (max-width: 600px) {
    .container { padding: 12px; }
    .bar-label { width: 70px; font-size: 11px; }
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🧬 PepScope <span>Multi-Function Classifier</span></h1>
    <p>Input a peptide sequence and predict 6 possible functions with confidence levels</p>
  </header>

  <div class="card">
    <form id="f">
      <textarea id="seqs" placeholder="Paste one or more peptide sequences (one per line)"></textarea>
      <div class="hint">Try examples:</div>
      <div class="examples">
        <span class="example-btn" onclick="ex('LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES')">LL-37 (human)</span>
        <span class="example-btn" onclick="ex('FLPLLAGLAANFLPTIICKISYKC')">Anticancer</span>
        <span class="example-btn" onclick="ex('GLFDVIKKVASVIGGL')">Antifungal</span>
        <span class="example-btn" onclick="ex('KRIVQRIKDFLR')">Antiviral</span>
      </div>
      <div class="row">
        <button class="primary" type="submit">Predict</button>
        <span id="spinner">Loading model...</span>
      </div>
    </form>
  </div>

  <div id="results"></div>
</div>

<footer>PepScope &middot; ESM-2 + GCN + Sequence + Fingerprint</footer>

<script>
const COLORS = ['#d32f2f','#388e3c','#1976d2','#7b1fa2','#f57c00','#00838f'];
const NAMES  = ['AntiCancer','AntiFungal','AntiGramPos','AntiGramNeg','AntiViral','AntiHypertensive'];
const SHORT  = ['Cancer','Fungal','Gram+','Gram-','Viral','Hypert.'];

function level(p, thr) {
  if (p >= 0.8) return 'strong';
  if (p >= thr) return 'medium';
  if (p >= 0.2) return 'weak';
  return 'none';
}

function render(r) {
  if (r.error) return '<div class="res-card"><div class="error-box">Warning: ' + r.seq + ': ' + r.error + '</div></div>';

  let items = NAMES.map((n,i) => ({name:n, short:SHORT[i], prob:r.probs[n], thr:r.thresholds[n], idx:i}));
  items.sort((a,b) => b.prob - a.prob);

  let tags = '';
  for (const it of items) {
    const l = level(it.prob, it.thr);
    tags += '<span class="tag tag-' + l + '">' + it.short + ' ' + (it.prob*100).toFixed(0) + '%</span>';
  }

  const top = items[0];
  const topL = level(top.prob, top.thr);
  const topStr = topL === 'strong' ? 'Most likely: ' : (topL === 'medium' ? 'Possible: ' : 'Uncertain: ');

  let bars = '';
  for (const it of items) {
    const pct = (it.prob * 100).toFixed(1);
    const thrPct = (it.thr * 100).toFixed(0);
    const isPos = it.prob >= it.thr;
    bars += '<div class="bar-row">'
          + '<div class="bar-label">' + it.short + '</div>'
          + '<div class="bar-track">'
          + '<div class="bar-fill" style="width:' + Math.max(+pct, 3) + '%;background:' + COLORS[it.idx % COLORS.length] + '">'
          + (+pct >= 12 ? pct + '%' : '') + '</div></div>'
          + '<div class="bar-pct">' + (isPos ? 'OK' : '') + ' <' + thrPct + '%</div></div>';
  }

  return '<div class="res-card">'
       + '<div class="res-seq">' + r.seq + '</div>'
       + '<div class="summary">' + tags + '</div>'
       + bars
       + '<div class="top-hit">' + topStr + '<strong>' + top.name + '</strong> (' + (top.prob*100).toFixed(1) + '%)</div>'
       + '</div>';
}

document.getElementById('f').onsubmit = async function(e) {
  e.preventDefault();
  const btn = document.querySelector('button.primary');
  const sp = document.getElementById('spinner');
  const res = document.getElementById('results');
  btn.disabled = true; sp.classList.add('active'); res.innerHTML = '';

  const raw = document.getElementById('seqs').value.trim();
  if (!raw) { btn.disabled = false; sp.classList.remove('active'); return; }

  try {
    const r = await fetch('/predict', {
      method:'POST',
      headers:{'Content-Type':'application/x-www-form-urlencoded'},
      body:'sequences=' + encodeURIComponent(raw),
    });
    const d = await r.json();
    if (d.error) { res.innerHTML = '<div class="card"><div class="error-box">' + d.error + '</div></div>'; }
    else {
      let html = '<div class="card"><h3 style="margin-bottom:14px;">Results</h3>';
      if (d.time) html += '<div style="font-size:12px;color:#888;margin-bottom:12px;">Time: ' + d.time.toFixed(2) + 's</div>';
      for (const r2 of d.results) html += render(r2);
      html += '</div>';
      res.innerHTML = html;
    }
  } catch(e) { res.innerHTML = '<div class="card"><div class="error-box">Request failed: ' + e.message + '</div></div>'; }

  btn.disabled = false; sp.classList.remove('active');
};

function ex(s) { document.getElementById('seqs').value = s; document.querySelector('button.primary').click(); }
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE


@app.post("/predict")
async def predict(sequences: str = Form(...)):
    lines = [s.strip() for s in sequences.strip().split("\n") if s.strip()]
    if not lines:
        return {"error": "No sequences provided"}
    t0 = time.time()
    try:
        predictor = get_predictor()
        results = predictor.predict(lines)
        return {"results": results, "time": round(time.time() - t0, 3)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("Starting PepScope Web App...")
    print("Open http://localhost:6006")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 6006)))

