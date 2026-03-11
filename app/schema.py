from pydantic import BaseModel


class WearableInput(BaseModel):
    steps: int
    resting_hr: int
    hrv: int
    sleep_hours: float
    active_minutes: int


class PredictionResponse(BaseModel):
    anomaly: bool
    anomaly_score: float
