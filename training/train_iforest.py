from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest


def main():
    data_path = Path("artifacts/synthetic_wearable_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: {data_path}. Önce make_synth_data.py çalıştırılmalı."
        )

    df = pd.read_csv(data_path)

    feature_cols = [
        "steps",
        "resting_hr",
        "hrv",
        "sleep_hours",
        "active_minutes"
    ]

    X = df[feature_cols].copy()

    model = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42
    )

    model.fit(X)

    df["anomaly_label"] = model.predict(X)
    df["anomaly_score"] = model.decision_function(X)

    predicted_anomalies = (df["anomaly_label"] == -1).sum()
    injected_anomalies = df["is_injected_anomaly"].sum()

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "iforest_model.joblib"
    scored_data_path = output_dir / "scored_wearable_data.csv"

    joblib.dump(model, model_path)
    df.to_csv(scored_data_path, index=False)

    print("Model eğitildi ve kaydedildi:")
    print(model_path)

    print("\nSkorlanmış veri kaydedildi:")
    print(scored_data_path)

    print("\nKullanılan feature'lar:")
    print(feature_cols)

    print("\nToplam satır sayısı:", len(df))
    print("Injected anomaly sayısı:", int(injected_anomalies))
    print("Modelin anomalik işaretlediği satır sayısı:", int(predicted_anomalies))

    print("\nİlk 10 skorlanmış satır:")
    print(df[[
        "user_id",
        "date",
        "steps",
        "resting_hr",
        "hrv",
        "sleep_hours",
        "active_minutes",
        "is_injected_anomaly",
        "anomaly_label",
        "anomaly_score"
    ]].head(10))


if __name__ == "__main__":
    main()
