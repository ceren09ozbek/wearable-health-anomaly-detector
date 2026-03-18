from fastapi import FastAPI
from app.schema import WearableInput, PredictionResponse
from app.model import predict_wearable_metrics


app = FastAPI(
    title="Wearable Anomaly Detection API",
    version="2.0"
)


@app.get("/")
def root():
    return {"message": "Wearable anomaly detection API with insight engine is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: WearableInput):

    result = predict_wearable_metrics(
        steps=data.steps,
        resting_hr=data.resting_hr,
        hrv=data.hrv,
        sleep_hours=data.sleep_hours,
        active_minutes=data.active_minutes
    )

    return PredictionResponse(
        anomaly=result["anomaly"],
        anomaly_score=result["anomaly_score"],
        observations=result["observations"],
        insight=result["insight"]
    )
