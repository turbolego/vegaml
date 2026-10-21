#!/usr/bin/env bash
# Deploy the vegaml WCAG demo backend to Modal using MODAL_TOKEN_ID / MODAL_TOKEN_SECRET.
#
# Auth: the Modal SDK reads MODAL_TOKEN_ID + MODAL_TOKEN_SECRET from the environment
# (https://modal.com/docs/sdk/py/latest/config) — no `modal setup`, no .modal.toml.
#
# Usage:
#   # 1) put your credentials in deploy/modal/.env (gitignored), OR export them:
#   export MODAL_TOKEN_ID=ak-...
#   export MODAL_TOKEN_SECRET=as-...
#   # 2) run:
#   bash deploy/modal/deploy.sh
#
# What it does: prewarm the model volume (one-off) → deploy the /v1/decide web
# endpoint → smoke-test /health and CORS → print the ?apiBase URL for the demo.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root

# Optional: load deploy/modal/.env if present (never committed).
if [[ -f deploy/modal/.env ]]; then
  set -a; source deploy/modal/.env; set +a
fi

command -v python3 >/dev/null
if [[ -z "${MODAL_TOKEN_ID:-}" || -z "${MODAL_TOKEN_SECRET:-}" ]]; then
  echo "ERROR: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are not set." >&2
  echo "  export them in this shell, or put them in deploy/modal/.env" >&2
  echo "  (see https://modal.com/docs/sdk/py/latest/config)" >&2
  exit 2
fi

# Modal CLI: use it if on PATH, else pip-install into a throwaway venv.
MODAL_CLI="${MODAL_CLI:-}"
if [[ -z "$MODAL_CLI" ]]; then
  if command -v modal >/dev/null 2>&1; then
    MODAL_CLI="modal"
  else
    VENV=$(mktemp -d)/venv
    python3 -m venv "$VENV" && "$VENV/bin/pip" -q install modal
    MODAL_CLI="$VENV/bin/modal"
  fi
fi

echo "==> auth check (modal token info)"
"$MODAL_CLI" token info || { echo "auth failed — check the token pair" >&2; exit 1; }

echo "==> prewarming model volume (one-off; skips if already warmed)"
"$MODAL_CLI" run deploy/modal/app.py::prewarm

echo "==> deploying web endpoint"
DEPLOY_OUT=$("$MODAL_CLI" deploy deploy/modal/app.py)
echo "$DEPLOY_OUT"
URL=$(echo "$DEPLOY_OUT" | grep -oE "https://[a-z0-9.-]*modal\.host[^ ]*" | head -1 || true)

if [[ -n "${URL:-}" ]]; then
  echo
  echo "==> smoke test $URL/health (first call boots the container: ~1-2 min)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    H=$(curl -s -m 120 "$URL/health" || true)
    [[ "$H" == *"model_loaded":true* ]] && break
    sleep 20
  done
  echo "health: $H"
  echo "CORS preflight:"
  curl -s -i -X OPTIONS "$URL/v1/decide" \
    -H "Origin: https://turbolego.github.io" -H "Access-Control-Request-Method: POST" \
    | grep -iE "^HTTP|^access-control" || true
  echo
  echo "Use in the demo page:"
  echo "  ${URL%/}/   →   https://turbolego.github.io/vegaml/wcag-demo/?apiBase=${URL%/}"
else
  echo "deployed; no URL parsed from output — check the URL above and the Modal dashboard."
fi
