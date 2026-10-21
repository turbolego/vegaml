# Host the vegaml WCAG demo backend on Modal

`modal deploy` gives us a public HTTPS endpoint that the static demo page
(`docs/wcag-demo/`, hosted on GitHub Pages) can call **directly** — no own
machine, no mixed-content block (the endpoint is HTTPS), and CORS is enabled.

```
docs/wcag-demo (static, GitHub Pages)   →   ?apiBase=https://<modal-host>   →   this Modal app (vegaml-serve FastAPI, CPU, scale-to-zero)
```

## Why Modal over the other options

| Provider | Idle cost | Wake | Notes |
|----------|-----------|------|-------|
| **Modal** | **$0** (scale-to-zero) | ~1 min (model from Volume) | Pay-per-second; $30/mo free credit on Starter |
| RunPod T4 | pod stopped = $0 | ~1–2 min boot | GPU, fastest inference; per-hour billing only while running |
| Fly.io | ~$16–17/mo always-on (or auto-suspend) | ~5 min | int8 model to fit 4 GB |
| HF Space | free hardware | ~1 min cold load on wake | 2026: a *paid* account (~$9/mo Pro) is required to create a compute Space |
| Vercel | — | — | **Not suitable**: serverless, no persistent process, every cold start re-loads torch + model, CPU-only |

Modal wins on "as few resources as possible": $0/mo plan, $0 while idle, and a
persistent Volume removes the ~7-minute model download from every cold start.

## Setup (once)

On any machine with a [Modal account](https://modal.com) (Starter plan:
$0/mo + $30/mo free compute credit):

```bash
pip install modal
# credentials: MODAL_TOKEN_ID + MODAL_TOKEN_SECRET (env vars or deploy/modal/.env)
```

## Deploy

```bash
bash deploy/modal/deploy.sh
```

The runner authenticates with the `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` env
vars (or `deploy/modal/.env`, gitignored) — no `modal setup` — then prewarms
the volume, deploys, smoke-tests `/health` + CORS, and prints the `?apiBase=`
URL. Or step by step:

```bash
modal run deploy/modal/app.py::prewarm   # one-off: cache the model in the Volume
modal deploy deploy/modal/app.py          # deploys the /v1/decide endpoint
```

## Use it

Open the hosted demo with the endpoint as the API base:

```
https://turbolego.github.io/vegaml/wcag-demo/?apiBase=https://<workspace>-vegaml-wcag-demo.<region>.modal.host
```

- First call after idle: ~60–90 s (container boot + model load from the Volume).
- Subsequent calls while warm: single question ≈ 1 s; the full 63-question
  batch ≈ 15–30 min on 4 vCPU.
- Want the full batch in seconds? Edit `SPECS` in `deploy/modal/app.py` to add
  `gpu="A10G"` (≈ $1.10/hr while running, ~$0.02 per batch).

## Cost

- Idle: **$0** (scaled down after ~10 idle minutes).
- Prewarm (one-time): a few cents.
- A 63-question demo session: ~$0.10–0.20 on CPU (covered by the $30/mo free
  credit, i.e. ~150 sessions/month).
