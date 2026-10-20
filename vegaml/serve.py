"""vegaml-serve: the decide() call over HTTP, one process, one loaded checkpoint.

    pip install "vegaml[serve]"
    vegaml-serve --model 4b --port 8000

    curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
        "state": "I was billed twice, refund me or I cancel.",
        "questions": {"churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"}}}'

POST /v1/decide answers one state; POST /v1/decide/batch answers a list on the same loaded
checkpoint, in order. GET /health says whether the model is up; GET /v1/model says what is
loaded. Responses are exactly what decide() returns, so a client written against the Python
API reads the same fields over HTTP.

One checkpoint, one process: requests are serialized around the model with a lock, because a
decision is a single forward pass measured in tens of milliseconds, and two interleaved ones
would share mutable per-call state. Scale-out is more processes behind any HTTP balancer,
not threads in this one.
"""
from __future__ import annotations

import argparse
import threading
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class DecideRequest(BaseModel):
    state: Any
    questions: Dict[str, Dict[str, Any]]
    mode: Optional[Literal["auto", "engine", "ttt", "both"]] = None
    alpha: str = "0.1"
    views: int = Field(1, ge=1, le=3)


class BatchRequest(BaseModel):
    items: List[DecideRequest] = Field(..., min_length=1, max_length=256)


def create_app(model=None, model_name: Optional[str] = None, load_kw: Optional[Dict] = None,
               cors: bool = False, cors_origins: Optional[List[str]] = None) -> FastAPI:
    """The FastAPI app. Pass a loaded model (tests do), or a name and the first request loads it.

    cors=True adds CORSMiddleware so a static page hosted on another origin (e.g. the
    GitHub Pages demo) can call the local server directly. Default origins are "*".
    """
    app = FastAPI(title="vegaml", docs_url="/docs")
    app.state.model = model
    app.state.model_name = model_name
    app.state.load_kw = load_kw or {}
    app.state.lock = threading.Lock()

    if cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    def _model():
        with app.state.lock:
            if app.state.model is None:
                import vegaml
                app.state.model = vegaml.load(app.state.model_name, **app.state.load_kw)
        return app.state.model

    def _decide(req: DecideRequest) -> Dict:
        v = _model()
        with app.state.lock:
            try:
                return v.decide(req.state, req.questions, mode=req.mode, alpha=req.alpha, views=req.views)
            except (ValueError, RuntimeError) as e:
                # A bad question (unknown type, no fitted head in mode="ttt") is the caller's to fix.
                raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/health")
    def health() -> Dict:
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.get("/v1/model")
    def model_info() -> Dict:
        v = _model()
        eng = getattr(v, "engine", None)
        cfg = getattr(eng, "cfg", {}) or {}
        return {"model": app.state.model_name or "default",
                "backbone": cfg.get("backbone_name"), "max_len": cfg.get("max_len"),
                "mode": getattr(v, "mode", None), "heads_fitted": len(getattr(v, "heads", {}))}

    @app.post("/v1/decide")
    def decide(req: DecideRequest) -> Dict:
        return _decide(req)

    @app.post("/v1/decide/batch")
    def decide_batch(req: BatchRequest) -> Dict:
        return {"results": [_decide(item) for item in req.items]}

    return app


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="vegaml-serve", description="Serve decide() over HTTP.")
    ap.add_argument("--model", default=None,
                    help="checkpoint: nothing for the 0.8B, '4b', a Hub id or a path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--preload", action="store_true",
                    help="load the checkpoint before binding the port instead of on the first request")
    ap.add_argument("--cors", action="store_true",
                    help="add CORS headers so a page on another origin (e.g. the GitHub Pages demo) "
                         "can call this server; use with --host 0.0.0.0 or a reachable address")
    ap.add_argument("--cors-origins", default=None,
                    help="comma-separated allow-list for --cors (default: *)")
    args = ap.parse_args(argv)
    import uvicorn
    load_kw = {}
    if args.device:
        load_kw["device"] = args.device
    if args.max_len:
        load_kw["max_len"] = args.max_len
    model = None
    if args.preload:
        import vegaml
        model = vegaml.load(args.model, **load_kw)
    origins = [o.strip() for o in args.cors_origins.split(",")] if args.cors_origins else None
    app = create_app(model=model, model_name=args.model, load_kw=load_kw,
                     cors=args.cors, cors_origins=origins)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
