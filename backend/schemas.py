from pydantic import BaseModel, Field
from typing import List


class TLERecord(BaseModel):
    name: str
    norad_id: int
    line1: str
    line2: str


class TLEResponse(BaseModel):
    count: int
    records: List[TLERecord]


class PropagateRequest(BaseModel):
    line1: str
    line2: str
    minutes: int = Field(ge=1, le=24 * 60, default=360)


class State(BaseModel):
    t: float
    r: list[float]
    v: list[float]


class PropagateResponse(BaseModel):
    states: List[State]


class RiskTarget(BaseModel):
    name: str
    states: List[State]


class RiskRequest(BaseModel):
    debris: RiskTarget
    targets: List[RiskTarget]
    threshold_km: float = 5.0


class CloseApproach(BaseModel):
    target: str
    min_distance_km: float
    timestamp_min: float
    rel_speed_kms: float
    risk_score: float


class RiskResponse(BaseModel):
    approaches: List[CloseApproach]


class PredictRequest(BaseModel):
    line1: str
    line2: str
    minutes: int = 360
    use_baseline_if_missing: bool = True


class PredictResponse(BaseModel):
    states: List[State]
    source: str
