// vegaml WCAG 2.2 detector — the same question sheet as LayaWCAG, answered by
// vegaml's calibrated probabilities instead of Laya's trained linear probe.
//
// The page talks to a running `vegaml-serve` (or the same-origin /api proxy
// shipped here). Each of the 63 criteria is a boolean question:
//   p("True")  -> probability the markup violates that criterion
// The verdict is p(violated) >= threshold.
const $ = (id) => document.getElementById(id);

const EXAMPLES = {
  "Image without alt text": '<img src="chart.png" width="200">\n<p>Q3 results</p>',
  "Button with no accessible name": '<div role="button" class="btn"></div>',
  "Form input without label": '<form><input type="email" name="email" placeholder="you@example.com"><button>Go</button></form>',
  "Heading level skipped": '<h1>Page title</h1>\n<h4>First section</h4>\n<p>Body text</p>',
  "Link with no discernible text": '<a href="/next"><img src="arrow.gif"></a>',
  "Iframe without title": '<iframe src="https://example.com/widget" width="600" height="400"></iframe>',
  "Clean semantic snippet": '<main aria-label="Main">\n  <h1>Contact us</h1>\n  <label for="email">Email</label>\n  <input id="email" type="email" name="email">\n  <img src="logo.png" alt="Company logo">\n  <button type="submit">Send</button>\n</main>',
};

let criteria = [];        // [{criterion, principle, level, description, instructions}]
let lastReport = null;    // the full report we can download

// Default API base: the same-origin /api proxy when the page runs next to server.py,
// otherwise the hosted Modal backend (scale-to-zero; first call wakes it in ~1 min),
// overridable with ?apiBase=<url>.
const DEFAULT_REMOTE_API = "https://hummernkloa--api.us-east.modal.run";

function defaultServer() {
  const h = location.hostname;
  return (h === "localhost" || h === "127.0.0.1") ? "/api" : DEFAULT_REMOTE_API;
}

async function fetchJSON(url, opt) { const r = await fetch(url, opt); if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); }
function server() { return $("server").value.replace(/\/$/, ""); }

async function checkServer() {
  const s = $("serverStatus"); s.className = "status"; s.textContent = "Checking…";
  try {
    const h = await fetchJSON(server() + "/health");
    const m = await fetchJSON(server() + "/model").catch(() => null);
    s.className = h.model_loaded ? "status ok" : "status";
    if (h.model_loaded) {
      s.textContent = `vegaml ${m?.model || "default"} · reader ${m?.backbone || "?"} · context ${m?.max_len ?? "?"} tokens · ready`;
    } else {
      s.textContent = "Server up, model still loading — wait a minute and check again (cold load of the 0.8B reader takes several minutes).";
    }
  } catch (e) {
    s.className = "status err";
    s.textContent = `Cannot reach server (${e.message}). The backend may still be waking (first call after idle takes ~1 min) — retry Check server, or point the API input at your own vegaml-serve (run: vegaml-serve --model 0.8b --host 0.0.0.0 --port 8080 --preload --cors).`;
  }
}

function setStatus(msg, cls = "") { const s = $("status"); s.textContent = msg; s.className = "status " + cls; }

function buildQuestions() {
  // Criterion id -> boolean question whose instructions carry the intent.
  const q = {};
  for (const c of criteria) q[c.criterion] = { type: "boolean", instructions: c.instructions };
  return q;
}

async function analyze() {
  const html = $("html").value.trim();
  if (!html) { setStatus("Paste some HTML first.", "err"); return; }
  const th = parseFloat($("threshold").value) || 0.5;
  $("analyze").disabled = true;
  setStatus(`Asking vegaml 63 questions…`);
  const t0 = performance.now();
  try {
    const out = await fetchJSON(server() + "/decide", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ state: html, questions: buildQuestions() }),
    });
    const ms = Math.round(performance.now() - t0);
    lastReport = { html, threshold: th, latency_ms: ms, server: server(), criteria: [], raw: out };
    const answers = out.answers || {};
    const rows = criteria.map((c) => {
      const a = answers[c.criterion] || {};
      const p = parseFloat(a.boolean);          // p(violated) for a boolean question
      return {
        criterion: c.criterion, principle: c.principle, level: c.level, description: c.description,
        p_violated: Number.isFinite(p) ? p : null,
        abstain: !!a.abstain, unbound: !!a.unbound,
        prediction_set: a.prediction_set, confidence: a.confidence,
      };
    });
    rows.sort((x, y) => (y.p_violated ?? -1) - (x.p_violated ?? -1));
    lastReport.criteria = rows;
    const violations = rows.filter((r) => r.p_violated != null && r.p_violated >= th);
    render(rows, violations, out, ms);
    setStatus(`Done in ${ms} ms — ${violations.length} of ${rows.length} criteria at or above p ≥ ${th}.`, "ok");
  } catch (e) {
    setStatus(`decide() failed: ${e.message}`, "err");
  } finally {
    $("analyze").disabled = false;
  }
}

function bar(p) {
  const w = Math.max(0, Math.min(100, Math.round((p ?? 0) * 100)));
  return `<div class="bar"><i style="width:${w}%"></i></div>`;
}

function render(rows, violations, out, ms) {
  const s = $("summary");
  s.innerHTML =
    `<span class="pill">${violations.length} violation${violations.length === 1 ? "" : "s"} / ${rows.length}</span>` +
    `<span class="pill">p ≥ ${$("threshold").value}</span>` +
    `<span class="pill">${ms} ms</span>` +
    (out.n_tokens ? `<span class="pill">${out.n_tokens} tokens in</span>` : "");

  // Violation table
  const vbody = $("violationRows");
  vbody.innerHTML = violations.map((r) => `<tr>
      <td class="crit">${r.criterion}<span class="lv ${r.level.toLowerCase()}">${r.level}</span></td>
      <td>${r.description}</td>
      <td class="pct">${(r.p_violated * 100).toFixed(1)}%</td>
      <td>${bar(r.p_violated)} ${r.abstain ? '<span class="flag">abstain</span> ' : ""}${r.unbound ? '<span class="flag">OOD</span>' : ""}</td>
    </tr>`).join("") || '<tr><td colspan="4" class="muted">No criterion crossed the threshold.</td></tr>';
  $("violationWrap").style.display = violations.length ? "" : "none";

  // Full ranked table
  $("allRows").innerHTML = rows.map((r) => `<tr>
      <td class="crit">${r.criterion}<span class="lv ${r.level.toLowerCase()}">${r.level}</span></td>
      <td>${r.description}</td>
      <td class="pct">${r.p_violated == null ? "—" : (r.p_violated * 100).toFixed(1) + "%"}</td>
      <td>${r.p_violated == null ? "—" : bar(r.p_violated)}</td>
    </tr>`).join("");

  $("raw").textContent = JSON.stringify(out, null, 2);
  $("results").classList.remove("hidden");
  $("download").classList.remove("hidden");
}

function download() {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "wcag-vegaml-report.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

// --- wire up ---
async function init() {
  criteria = await fetchJSON(new URL("./wcag_criteria.json", location.href));
  $("server").value = new URLSearchParams(location.search).get("apiBase") || defaultServer();
  $("example").addEventListener("change", (e) => { const t = EXAMPLES[e.target.value]; if (t) $("html").value = t; });
  $("check").addEventListener("click", checkServer);
  $("analyze").addEventListener("click", analyze);
  $("download").addEventListener("click", download);
  $("html").value = EXAMPLES["Image without alt text"];
  $("analyze").disabled = false;
  checkServer();
}
document.addEventListener("DOMContentLoaded", init);
