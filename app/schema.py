from pydantic import BaseModel
from typing import List


class WearableInput(BaseModel):
    steps: int
    resting_hr: int
    hrv: int
    sleep_hours: float
    active_minutes: int


class PredictionResponse(BaseModel):
    anomaly: bool
    anomaly_score: float
    observations: List[str]
    insight: str
