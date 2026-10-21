# WCAG 2.2 demo: paste markup, score all 63 criteria with vegaml

A webpage demo in the spirit of [turbolego/LayaWCAG](https://github.com/turbolego/LayaWCAG)
(paste HTML → verdict per criterion → ranked probabilities → report JSON), but the brain is
**vegaml's HTTP API** instead of an in-browser ONNX probe. LayaWCAG runs the whole model in
the browser; vegaml is a server-side reader (frozen LLM + 14.3M decision engine), so the demo
is a thin web UI on top of `vegaml-serve`.

## Question sheet

`wcag_criteria.json` is the same 63 WCAG 2.2 criteria LayaWCAG ships
(`LayaWCAG/model/wcag_criteria.json`), rebuilt as vegaml boolean questions:

```json
{ "criterion": "1.1.1", "level": "A",
  "instructions": "Does this HTML markup fail WCAG 2.2 criterion 1.1.1 (non-text content),
                   i.e. it is violated because non-text content ... has no alt text? Answer from the markup alone." }
```

Each `instructions` string carries the criterion's intent so the zero-shot engine has
discriminative signal; the criterion id is the question key, so answers line up 1:1 with
the 63-row table. Regenerate after editing intents with `python3 make_criteria.py`.

## Run it

Two modes:

### A. Local (same-origin proxy — no CORS involved)

```bash
pip install "vegaml[serve]"

# 1) the model server (--preload loads before binding, so /health only goes green when ready)
vegaml-serve --model 0.8b --host 0.0.0.0 --port 8080 --preload &

# 2) this demo: static files + a same-origin /api proxy to the server above
python3 server.py
# -> http://127.0.0.1:8090
```

On `localhost`/`127.0.0.1` the page defaults its API base to `/api` (the proxy).

### B. GitHub Pages (this folder is under `docs/` so Pages can deploy it)

GitHub Pages only serves the repo root or `docs/`, so the demo lives at `docs/wcag-demo/`
and deploys to `https://turbolego.github.io/vegaml/wcag-demo/` (Pages setting: deploy the
`docs` folder). The page has no model of its own — it talks to a `vegaml-serve` you run
locally (or on a machine you can reach), so start the server with CORS:

```bash
vegaml-serve --model 0.8b --host 0.0.0.0 --port 8080 --preload --cors
```

On a non-localhost host the page defaults its API base to `http://127.0.0.1:8080`;
override with `?apiBase=<url>`. (The demo must remain static: Pages has no server, so the
reader + engine always run wherever *you* point the page.)

In both modes: open the page, click **Check server**, pick an example snippet or paste your
own markup, set a threshold (default 0.5), click **Analyze**. The page posts one
`/v1/decide` call with all 63 questions on the pasted state; results render as a violations
table + a full ranked list + the raw answer JSON, downloadable as `wcag-vegaml-report.json`.

## Troubleshooting: "Cannot reach server" on the hosted page

The GitHub Pages page is **HTTPS**; browsers block an HTTPS page from fetching an
**`http://`** backend (active mixed content). So `?apiBase=http://<any-ip>:8080` will fail
in the browser even when the server is up and `--cors` is set — the block is policy, not
connectivity. The options, in order of effort:

1. **Local mode (no frictions)** — mode A above: the page and the backend share an origin
   (`http://127.0.0.1:8090`), so nothing is blocked. This is the default dev workflow.
2. **Same-machine backend** — run `vegaml-serve ... --cors` on the machine whose browser
   opens the hosted page and keep the default `http://127.0.0.1:8080`; browsers allow
   `http://127.0.0.1` from an HTTPS page (potentially-trustworthy origin), unlike remote
   IPs.
3. **HTTPS backend on the tailnet** — on the backend box: `sudo tailscale serve --bg 8080`
   (needs sudo: the serve config is root-gated), then open the hosted page with
   `?apiBase=https://100.83.145.68` (the tailnet cert is trusted by tailnet clients).
4. **In-browser ONNX (roadmap)** — like LayaWCAG: no server at all. Needs the 0.8B reader
   exported to ONNX Runtime Web; `export_engine()` ships the 14.3M engine only.

## What you get (and what you don't)

- **Zero-shot off the shelf**: no WCAG training data, no fitted head. The probabilities are
  the vega-U engine's calibrated outputs; `abstain` / `unbound` flags in the raw JSON mark
  questions the engine considers out of distribution.
- **Not a substitute for axe-core**: the reader sees markup text only, like LayaWCAG.
  Contrast (1.4.3/1.4.6), reflow (1.4.10), and target-size (2.5.5) criteria need computed
  styles/screenshots to be conclusive; vegaml's vision path (Qwen3.5 readers, image state)
  is where that goes — see `vegaml/_vision.py`.
- **Calibrate on your own labels** to make the verdicts domain-honest:
  collect `{"state": "<markup>", "violated": true}` pairs and run `vegaml calibrate`
  (see `docs/calibration.md`); the server then serves the refit temperature/conformal sets.

## Files

| File | What it is |
|---|---|
| `index.html` | the page: server check, paste box, results, raw JSON, report download |
| `app.js` | the logic: builds the 63-question sheet, calls `/decide`, renders |
| `server.py` | stdlib-only host: static files + `/api` → `vegaml-serve` proxy (no CORS needed) |
| `wcag_criteria.json` | 63 criteria as vegaml boolean questions (generated by `make_criteria.py`) |
| `make_criteria.py` | rebuilds the question sheet from LayaWCAG's criterion list + hand-written intents |
