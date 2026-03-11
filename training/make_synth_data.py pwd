from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_wearable_data(
    n_users: int = 100,
    n_days: int = 30,
    anomaly_ratio: float = 0.08,
    seed: int = 42
):
    rng = np.random.default_rng(seed)

    rows = []
    start_date = datetime(2026, 1, 1)

    for user_idx in range(1, n_users + 1):

        user_id = f"user_{user_idx:03d}"

        base_steps = rng.integers(5000, 10000)
        base_resting_hr = rng.integers(58, 72)
        base_hrv = rng.integers(35, 65)
        base_sleep = rng.uniform(6.5, 8.5)
        base_active = rng.integers(30, 75)

        for day_idx in range(n_days):

            current_date = start_date + timedelta(days=day_idx)

            steps = int(np.clip(rng.normal(base_steps, 1800), 500, 20000))
            resting_hr = int(np.clip(rng.normal(base_resting_hr, 5), 45, 110))
            hrv = int(np.clip(rng.normal(base_hrv, 10), 5, 120))
            sleep_hours = float(np.clip(rng.normal(base_sleep, 0.9), 2.0, 12.0))
            active_minutes = int(np.clip(rng.normal(base_active, 15), 0, 180))

            is_injected_anomaly = 0

            if rng.random() < anomaly_ratio:

                is_injected_anomaly = 1

                anomaly_type = rng.choice([
                    "low_steps_high_hr",
                    "very_low_sleep",
                    "low_hrv_combo",
                    "inactive_day"
                ])

                if anomaly_type == "low_steps_high_hr":
                    steps = int(rng.integers(200, 1200))
                    resting_hr = int(rng.integers(85, 105))
                    active_minutes = int(rng.integers(0, 15))

                elif anomaly_type == "very_low_sleep":
                    sleep_hours = float(rng.uniform(2.5, 4.5))
                    resting_hr = int(rng.integers(80, 98))
                    hrv = int(rng.integers(10, 25))

                elif anomaly_type == "low_hrv_combo":
                    hrv = int(rng.integers(8, 20))
                    resting_hr = int(rng.integers(82, 100))
                    steps = int(rng.integers(500, 3000))

                elif anomaly_type == "inactive_day":
                    steps = int(rng.integers(100, 1500))
                    active_minutes = int(rng.integers(0, 10))
                    sleep_hours = float(rng.uniform(4.0, 5.5))

            rows.append({
                "user_id": user_id,
                "date": current_date.strftime("%Y-%m-%d"),
                "steps": steps,
                "resting_hr": resting_hr,
                "hrv": hrv,
                "sleep_hours": round(sleep_hours, 2),
                "active_minutes": active_minutes,
                "is_injected_anomaly": is_injected_anomaly
            })

    df = pd.DataFrame(rows)

    return df


def main():

    df = generate_wearable_data()

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "synthetic_wearable_data.csv"

    df.to_csv(output_path, index=False)

    print("Dataset oluşturuldu:")
    print(output_path)

    print("\nToplam satır sayısı:", len(df))

    print("\nİlk 5 satır:")
    print(df.head())

    anomaly_count = df["is_injected_anomaly"].sum()

    print("\nInjected anomaly sayısı:", anomaly_count)


if __name__ == "__main__":
    main()
