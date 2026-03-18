from pathlib import Path
import joblib
import numpy as np

from app.insight_engine import build_insight_payload
from app.llm_engine import generate_llm_insight


MODEL_PATH = Path("artifacts/iforest_model.joblib")

model = joblib.load(MODEL_PATH)


def predict_wearable_metrics(steps, resting_hr, hrv, sleep_hours, active_minutes):

    X = np.array([[steps, resting_hr, hrv, sleep_hours, active_minutes]])

    prediction = model.predict(X)[0]
    score = model.decision_function(X)[0]

    anomaly = prediction == -1

    # 1️⃣ observation üret (rule-based)
    insight_payload = build_insight_payload(
        steps=steps,
        resting_hr=resting_hr,
        hrv=hrv,
        sleep_hours=sleep_hours,
        active_minutes=active_minutes,
        anomaly=anomaly
    )

    # 2️⃣ LLM ile final insight üret
    llm_text = generate_llm_insight(
        observations=insight_payload["observations"],
        anomaly=anomaly
    )

    return {
        "anomaly": anomaly,
        "anomaly_score": float(score),
        "observations": insight_payload["observations"],
        "insight": llm_text
    }
