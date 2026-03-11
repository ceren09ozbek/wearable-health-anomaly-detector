from fastapi import FastAPI
from app.schema import WearableInput, PredictionResponse
from app.model import WearableAnomalyModel


app = FastAPI(
    title="Wearable Anomaly Detection API",
    version="1.0"
)

model = WearableAnomalyModel()


@app.get("/")
def root():
    return {"message": "Wearable anomaly detection API çalışıyor"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: WearableInput):

    anomaly, score = model.predict(
        steps=data.steps,
        resting_hr=data.resting_hr,
        hrv=data.hrv,
        sleep_hours=data.sleep_hours,
        active_minutes=data.active_minutes
    )

    return PredictionResponse(
        anomaly=anomaly,
        anomaly_score=score
    )
