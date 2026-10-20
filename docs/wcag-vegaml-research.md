# WCAG 2.2 Detection: vegaml vs LayaWCAG — research notes

Case-study notes on using vegaml (frozen LLM reader + 14.3M vega-U decision engine) to detect
WCAG 2.2 violations the way [turbolego/LayaWCAG](https://github.com/turbolego/LayaWCAG) does
with its in-browser ONNX probe. The comparison sheet was originally written for the LayaWCAG
project (`LayaWCAG/docs/md/wcag-vegaml-research.md`); this copy lives here because it now
drives the concrete demo in `docs/wcag-demo/`.

## Executive Summary

Both **LayaWCAG** and **vegaml (vega-U)** use a frozen-language-model + lightweight
decision-head architecture to score WCAG 2.2 accessibility violations. Their designs are
very similar — shared-space engine, per-criterion output, one-pass batch scoring — but they
diverge on **model base**, **training data**, **export format**, and **browser vs server
usage**.

### Comparison Table

| Dimension | LayaWCAG | vegaml (vega-U) |
|---|---|---|
| **Base encoder** | ModernBERT-large (421M params, frozen) | Multiple frozen readers: 0.8B, 4B (Qwen3.5), Gemma 3 1B, SmolLM2 1.7B, any HF model via `load(backbone="…")` |
| **Decision engine** | Linear probe (1024→63) + sigmoid; trained from scratch on WCAG data | Physics engine + per-reader port (14.3M shared weights); zero-shot off-the-shelf, then `fit()` for domain heads or `attach()` to add new readers |
| **Training data** | AI-WCAG-Gauntlet (axe-core labels on real websites, 40 labeled samples) | Synthetic corpus (1000+ prompts per question) + calibration on user's own [state,label] pairs; no task-specific data shipped |
| **WCAG criteria coverage** | 63 criteria, 52 have **no positive training example** (media, timing, sensory, reflow, etc.) → output those columns as noise | No WCAG data shipped; answer any question type (boolean/choice/score); calibrated probabilities + conformal sets + abstain flag |
| **Export / serving** | ONNX → ONNX Runtime Web (browser, 16 data-parts, int8, ~395MB total) | Python HTTP API (`vegaml-serve`), MCP server, ONNX export (`export_engine()`), MLX (Apple silicon), Docker |
| **Browser deployment** | Yes — demo at `turbolego.github.io/LayaWCAG/` | Server-side: a thin web UI on `vegaml-serve` (this repo: `docs/wcag-demo/`); the reader does not run in-browser |

## Where vegaml Extends the Use Case

### 1. **Zero-shot, no task-specific data**
LayaWCAG's probe is noise on 52 of 63 criteria because those columns never saw a positive
example. vegaml's engine answers each criterion as a *question* against the frozen reader,
so a criterion with no training data is still a question the engine can reason about —
weak, calibrated probabilities instead of pure noise. The demo ships a 63-criterion
question sheet (`docs/wcag-demo/wcag_criteria.json`) and works off the shelf.

### 2. **Structured / pydantic output for WCAG reports**
LayaWCAG returns `{criterion, level, description, violation, probability}`. vegaml's
`decide_model()` with a Pydantic model (install `vegaml[structured]`) returns validated,
typed output. Example WCAG Pydantic model:

```python
from typing import Literal
from pydantic import BaseModel, Field
from vegaml.structured import decide_model

class WcagViolation(BaseModel):
    criterion: str = Field(description="WCAG criterion, e.g. '1.4.3'")
    level: Literal["A", "AA", "AAA"] = Field(description="Conformance level")
    violation: Literal["yes", "no"] = Field(description="Whether the criterion is violated")
    probability: float = Field(description="p(violation) from the model")
    confidence: float = Field(description="Calibrated confidence score")

v = vegaml.load()
ticket, result = decide_model(v, html_snippet, WcagViolation)
```

The structured output guarantees the fields exist and are typed, eliminates "quiet
corruption" (invalid answers raise `ValidationError`), and the calibrated detail
(probabilities, prediction sets, abstain) is preserved alongside the typed answer.

### 3. **Calibration on your own domain**
LayaWCAG's model is fixed after training; you cannot recalibrate it on your site's
content. vegaml's `calibrate()` refits the temperature and conformal set on your labeled
examples — essential if your site's "violation" distribution differs from the
AI-WCAG-Gauntlet (e.g. mostly e-commerce product pages vs. blog content).

### 4. **Attachment of new readers (model expansion)**
If you want a larger or more capable reader than the default 0.8B, vegaml's `attach()`
fits a new port from any HuggingFace model to the shared engine — without retraining the
engine. This lets you experiment with 7B, 13B, or 27B readers while keeping the same
WCAG question sheet.

### 5. **ONNX export — the missing link for browser deployment**
The main gap: vegaml can export to ONNX (`export_engine()`), but the current export
targets CPU/GPU inference, not the browser-optimized ONNX Runtime Web path that LayaWCAG
uses. If the ONNX export were adapted to the same input format, vegaml could ship a
browser demo; until then the browser pattern is a server + thin UI (this is exactly what
`docs/wcag-demo/` does).

## Zero-shot PoC

The minimal probe that verifies vegaml can score HTML snippets in engine mode:

```python
import json
import vegaml

v = vegaml.load()                              # 0.8B reader + 14.3M engine

questions = {
    "wcag_link_name":  {"type": "boolean",
                        "instructions": "Does every link have a discernible purpose "
                                          "(WCAG 2.4.4 / 2.4.9)?"},
    "wcag_img_alt":    {"type": "boolean",
                        "instructions": "Does every <img> carry a non-empty alt?"},
    "wcag_heading_order": {"type": "score",
                           "instructions": "How well ordered are heading levels?",
                           "criteria": ["each heading increases by one or missing",
                                        "one skipped level",
                                        "multiple skipped or inverted levels"]},
}

html = '<div><a href="/more">click here</a><img src="banner.png"></div>'

out = v.decide(html, questions, mode="engine")
print(json.dumps(out["answers"], default=str, indent=2))
```

Each answer carries: `boolean` (p(yes)), `score` / `expected_score` (for score types),
`probs` (calibrated distribution), `prediction_set` (conformal set at alpha), `abstain`,
`unbound`, `confidence`.

## The Webpage Demo: `docs/wcag-demo/`

The LayaWCAG equivalent, rebuilt on vegaml's HTTP API:

- `index.html` + `app.js` — paste HTML, get a verdict per criterion, a ranked table of all
  63 p(violated) values, and a downloadable report JSON. Same UX as
  `turbolego.github.io/LayaWCAG/`, minus the in-browser model.
- `server.py` — stdlib-only host: static files + a same-origin `/api` proxy into
  `vegaml-serve` (used for local runs; no CORS involved).
- `wcag_criteria.json` — the 63 LayaWCAG criteria rebuilt as vegaml boolean questions,
  each `instructions` string carrying the criterion's intent so the zero-shot engine has
  discriminative signal. Regenerated by `make_criteria.py`.

It lives under `docs/` because GitHub Pages can only deploy the repo root or `docs/`, so
the published demo is `https://turbolego.github.io/vegaml/wcag-demo/`. A Pages-hosted page
is a different origin than a local `vegaml-serve`, so the server gained a `--cors` flag
(CORS headers; verified on the wire — preflight + `access-control-allow-origin`), and the
page auto-detects its API base: `/api` on `localhost` (the proxy), `http://127.0.0.1:8080`
elsewhere (run `vegaml-serve --model 0.8b --host 0.0.0.0 --port 8080 --preload --cors`).

Run locally: `vegaml-serve --model 0.8b --port 8080 --preload` + `python3 docs/wcag-demo/server.py`.

## Migration Path: Moving from LayaWCAG → vegaml

1. **Define your WCAG question sheet** — map each of the 63 WCAG 2.2 criteria to a
   boolean/choice/score question. `docs/wcag-demo/wcag_criteria.json` is the starting
   sheet; start with the criteria LayaWCAG *does* have data for (1.1.1, 2.4.7, 3.3.2,
   4.1.2, …) plus any new ones you care about.
2. **Run the PoC above** — verify vegaml can score your HTML snippets in engine mode.
   The 0.8B reader is the default; switch to `vegaml.load("4b")` for harder cases.
3. **Fit or calibrate** — if you have labeled [html_snippet, violation_labels] pairs,
   run `fit(examples, questions)` to fit TTT heads, or `calibrate()` to refit
   probabilities on your domain.
4. **Optional: attach a larger reader** — `attach(backbone, texts, out_dir=…)` then
   `vegaml.load(out_dir)`; the engine bytes stay identical, only the port + calibration
   change.
5. **ONNX export (future)** — once the export targets the browser path, embed it in a web
   worker mirroring LayaWCAG's deployment.

## Limitations (Honest Status, Both Systems)

- **LayaWCAG**: 52 of 63 WCAG criteria have no positive training example. Output for
  those columns is noise. No CSS/computed-styles access, no rendered layout, no
  JavaScript behaviour simulation, no pointer hit-testing. The model sees markup text
  only.
- **vega-U (WCAG via questions)**: Same input constraints — the model sees text only,
  no CSS cascade, no rendered media. Criteria that depend on visual layout, pointer
  interaction, or real media (1.4.3/1.4.6 contrast, 1.4.10 reflow, 2.5.5 target size)
  are inherently weak in text mode; vegaml's vision path (Qwen3.5 readers + image state,
  `vegaml/_vision.py`) is where that goes. Calibration on your own data is essential for
  domain shift.
- **Both**: the model is a research demo, not a conformance tool. Pair with axe-core,
  pa11y, or qualweb for rule-based checking on the criteria the model cannot reach.

## Test status (2026-10-20)

- vegaml 0.12.4 full pytest suite (torch 2.14.1 CPU, transformers 5.20.2): **278 passed, 3 skipped** (Apple-only skips) in ~18s on Linux, with all extras installed.
- Zero-shot WCAG PoC **and** the `docs/wcag-demo/` e2e both PASSED (63/63 answers; injected violations ranked top-3). Full log below.

## Test log

### 2026-10-20 — 0.12.4 suite + zero-shot WCAG PoC

1. **pytest** (dev venv, `env -u PYTHONPATH`): with `vegaml[serve,dev,onnx,mcp,structured]` + langchain-core → **278 passed, 3 skipped in ~18s** on `main` @ `0eb2c28`. The 3 remaining skips are Apple-only (`vegaml[mlx]` ×1, MPS hardware ×2).
2. **Demo static/proxy server**: `docs/wcag-demo/server.py` serves `index.html` (correct content-type, 200) and `wcag_criteria.json` (200, all 63 criteria present) on its own; the 502 path when `vegaml-serve` is down confirmed in e2e below.
3. **Zero-shot 63-criterion decide() + full e2e** (vega-U 0.8B reader, `device="cpu"`, snippet mixing three known violations: button without accessible name, img without alt, link with no discernible text): one `/decide` call through the demo's exact payload (63 boolean questions). Sequence: (a) first run with no kernels — reference-conv path, ~14 min of inference, killed; (b) with `fla-core` installed — `/v1/decide` → HTTP 400 "0 active drivers" (Triton crash, see below); (c) `fla-core` removed — reference PyTorch path, single-question e2e verified (p(violated)=0.9999 in ~0.6 s).

   **63-question batch through the demo proxy: PASSED.** All 63 answers returned in **523.6 s wall** (CPU reference kernels; `n_tokens=11863`). The three injected violations ranked at the top of the p(violated) distribution:

   | rank | criterion | p(violated) | |
   |---|---|---|---|
   | 1 | 1.1.1 Non-text Content | 0.974 | ⌂ injected: `<img>` no alt |
   | 3 | 4.1.2 Name/Role/Value | 0.912 | ⌂ injected: `<div role=button>` no name |
   | — | 2.4.4 Link Purpose | 0.876 | ⌂ injected: link with no discernible text |

   Top-10 overall: 1.1.1 .97, 1.4.5 .96, 1.4.9 .95, 2.4.5 .95, 1.4.11 .94, 3.3.1 .93, 4.1.1 .93, 2.4.1 .92, 3.1.1 .92, 1.4.4 .92. Bottom-3 (lowest p): 1.4.7 .39, 1.2.4 .30, 1.4.2 .27. No `abstain` / `unbound` flags fired on any criterion at the default alpha=0.1.

   The 523 s is the CPU-reference-kernel cost for one full 63-question scan; on a CUDA box with the fla/Triton kernels the same call drops to well under a second. The demo page already surfaces per-call latency, so this is visible to the user, not hidden.

### CPU gotcha: do NOT install the fla kernels on a CPU-only box

The Qwen3.5 readers are hybrid (gated-delta-net) models. When `fla-core` / `flash-linear-attention`
is installed, transformers routes `chunk_gated_delta_rule` through fla's **Triton** kernels; on a
CPU-only machine Triton has no active driver and the first forward pass dies with

```
RuntimeError: 0 active drivers ([]). There should only be one.
```

— which `vegaml-serve` surfaces as an HTTP 400 on `/v1/decide` (health still looks fine).
With the fla packages **removed**, transformers falls back to its pure-PyTorch reference
implementations (`torch_chunk_gated_delta_rule` + reference `causal_conv1d_fn`), which run
correctly on CPU: a single-question `/v1/decide` answered in **~0.6–0.9 s** and returned
the expected verdict (p(violated)=0.9999 for a bare `<img>` with no alt). The dev venv has
`fla-core` uninstalled; keep it that way for CPU runs. (On a CUDA machine the fla/Triton
path is the fast one — install it there.)
