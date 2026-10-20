# HTTP API

```bash
pip install "vegaml[serve]"
vegaml-serve --model 4b --port 8000          # --preload to load before binding
```

`--cors` adds CORS headers so a page hosted on another origin (e.g. the GitHub Pages demo
in `docs/wcag-demo/`) can call the server directly; `--cors-origins` limits the allowed
origins (default `*`). Without the flag the server stays same-origin-only.

## Endpoints

| Route | What it does |
| --- | --- |
| `POST /v1/decide` | `{state, questions, mode?, alpha?, views?}` in, the decide() dictionary out |
| `POST /v1/decide/batch` | `{"items": [...]}`, answered in order on the one loaded checkpoint |
| `GET /health` | `{"status": "ok", "model_loaded": bool}` |
| `GET /v1/model` | what is loaded: backbone, context limit, mode, fitted heads |
| `GET /docs` | the interactive schema |

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "state": "I was billed twice, refund me or I cancel.",
  "questions": {"churn": {"type": "boolean",
                          "instructions": "Does the customer threaten to leave?"}}}'
```

The response is exactly what the Python API returns, so client code reads the same fields
either way. A malformed request is a 422 naming the field; a bad question (unknown type,
`mode="ttt"` without a fitted head) is a 400 carrying the model's message.

## Concurrency

One process holds one checkpoint and serializes decisions around it; a decision is a single
forward pass measured in tens of milliseconds, so the lock costs less than making per-call
state concurrent-safe would. Scale out with more processes behind any HTTP balancer.
