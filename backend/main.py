from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .settings import settings
from . import client
from .schemas import (
    TLEResponse,
    TLERecord,
    PropagateRequest,
    PropagateResponse,
    State,
    RiskRequest,
    RiskResponse,
    CloseApproach,
    PredictRequest,
    PredictResponse,
)
from .propagation import propagate_tle, ECIState
from .risk import compute_close_approaches
import numpy as np
from fastapi import Query
from typing import Optional

app = FastAPI(title="Orbital Debris PINN API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/tle/{norad_id}", response_model=TLEResponse)
async def get_tle(norad_id: int):
    try:
        payload = await client.fetch_tle_by_norad(norad_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NASA API error: {e}")
    records = []
    if isinstance(payload, dict) and "member" in payload:
        items = payload["member"]
    elif isinstance(payload, dict) and "records" in payload:
        items = payload["records"]
    else:
        items = [payload] if isinstance(payload, dict) else payload

    for item in items:
        name = item.get("NAME") or item.get("name") or f"NORAD-{norad_id}"
        line1 = item.get("TLE_LINE1") or item.get("line1")
        line2 = item.get("TLE_LINE2") or item.get("line2")
        if not (line1 and line2):
            rec = item.get("tle") or {}
            line1 = line1 or rec.get("line1")
            line2 = line2 or rec.get("line2")
        if line1 and line2:
            records.append(
                TLERecord(name=name, norad_id=norad_id, line1=line1, line2=line2)
            )
    if not records:
        raise HTTPException(status_code=404, detail="No TLE found for NORAD ID")
    return TLEResponse(count=len(records), records=records)


@app.post("/api/propagate", response_model=PropagateResponse)
def post_propagate(req: PropagateRequest):
    states = propagate_tle(req.line1, req.line2, req.minutes)
    return PropagateResponse(
        states=[State(t=s.t, r=s.r.tolist(), v=s.v.tolist()) for s in states]
    )


def _to_states(out):
    return [
        State(
            t=float(o["t"]), r=[float(x) for x in o["r"]], v=[float(x) for x in o["v"]]
        )
        for o in out
    ]


from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "ml" / "checkpoints" / "model.pt"


@app.post("/api/predict", response_model=PredictResponse)
def post_predict(req: PredictRequest):
    if MODEL_PATH.exists():
        try:
            import torch, numpy as np

            model = torch.jit.load(str(MODEL_PATH))
            model.eval()
            base = propagate_tle(req.line1, req.line2, req.minutes)
            feats = []
            for s in base:
                feats.append(np.concatenate([s.r, s.v]))
            x = torch.tensor(np.stack(feats), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                delta = model(x).squeeze(0).numpy()
            out = []
            for i, s in enumerate(base):
                r = (s.r + delta[i, :3]).tolist()
                v = (s.v + delta[i, 3:]).tolist()
                out.append(dict(t=s.t, r=r, v=v))
            return PredictResponse(states=_to_states(out), source="pinn")
        except Exception as e:
            if not req.use_baseline_if_missing:
                raise HTTPException(
                    status_code=500, detail=f"PINN inference error: {e}"
                )
    states = propagate_tle(req.line1, req.line2, req.minutes)
    return PredictResponse(
        states=[State(t=s.t, r=s.r.tolist(), v=s.v.tolist()) for s in states],
        source="sgp4",
    )


@app.post("/api/risk", response_model=RiskResponse)
def post_risk(req: RiskRequest):
    debris_states = [
        ECIState(t=s.t, r=np.array(s.r), v=np.array(s.v)) for s in req.debris.states
    ]
    targets = {
        tgt.name: [
            ECIState(t=s.t, r=np.array(s.r), v=np.array(s.v)) for s in tgt.states
        ]
        for tgt in req.targets
    }
    results = compute_close_approaches(debris_states, targets, req.threshold_km)
    return RiskResponse(approaches=[CloseApproach(**r) for r in results])

@app.get("/api/debris/catalog", response_model=TLEResponse)
async def get_debris_catalog(
    name: str = Query(..., description="One of: fengyun1c, cosmos1408, iridium33"),
    limit: int = Query(25, ge=1, le=200),
):
    """
    Proxy/normalize popular debris catalogs from CelesTrak into TLEResponse shape.
    """
    try:
        data = await client.fetch_celestrak_catalog(name=name, limit=limit)
        return TLEResponse(**data)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load catalog: {e!s}")

@app.get("/api/debris/catalog", response_model=TLEResponse)
async def get_debris_catalog(name: str, limit: int = 25):
    try:
        data = await client.fetch_celestrak_catalog(name=name, limit=limit)
        return TLEResponse(**data)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))