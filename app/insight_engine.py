def generate_observations(steps, resting_hr, hrv, sleep_hours, active_minutes):
    observations = []

    if resting_hr >= 85:
        observations.append("Resting heart rate appears elevated")

    if hrv <= 20:
        observations.append("HRV is lower than expected")

    if sleep_hours <= 5:
        observations.append("Sleep duration is lower than expected")

    if steps <= 2000:
        observations.append("Daily step count is low")

    if active_minutes <= 15:
        observations.append("Daily activity level is low")

    return observations


def generate_insight(observations, anomaly):
    if not observations and not anomaly:
        return (
            "Your recent wearable metrics look stable overall. "
            "Your activity, sleep, and recovery-related signals appear to be within a generally expected range."
        )

    if not observations and anomaly:
        return (
            "Your wearable metrics show an unusual pattern compared to the expected range. "
            "Although no single signal stands out strongly, monitoring your recent routine may still be helpful."
        )

    joined = ", ".join(observations).lower()

    return (
        f"Your recent wearable metrics suggest that {joined}. "
        "This may reflect reduced recovery, increased stress load, or lower daily activity. "
        "Paying attention to sleep regularity, hydration, movement, and overall recovery may be helpful. "
        "If this pattern continues, consider consulting a healthcare professional."
    )


def build_insight_payload(steps, resting_hr, hrv, sleep_hours, active_minutes, anomaly):
    observations = generate_observations(
        steps=steps,
        resting_hr=resting_hr,
        hrv=hrv,
        sleep_hours=sleep_hours,
        active_minutes=active_minutes
    )

    insight = generate_insight(observations, anomaly)

    return {
        "observations": observations,
        "insight": insight
    }
