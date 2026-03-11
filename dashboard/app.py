import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Wearable Anomaly Dashboard", layout="wide")

st.title("Wearable Health Anomaly Dashboard")
st.caption("Synthetic wearable data + Isolation Forest anomaly detection")

data_path = Path("artifacts/scored_wearable_data.csv")

if not data_path.exists():
    st.error("scored_wearable_data.csv bulunamadı. Önce training/train_iforest.py çalıştırılmalı.")
    st.stop()

df = pd.read_csv(data_path)

total_records = len(df)
anomaly_count = int((df["anomaly_label"] == -1).sum())
anomaly_ratio = anomaly_count / total_records if total_records > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{total_records}")
col2.metric("Anomalies", f"{anomaly_count}")
col3.metric("Anomaly Ratio", f"{anomaly_ratio:.2%}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Anomaly Score Distribution")
    fig_hist = px.histogram(
        df,
        x="anomaly_score",
        nbins=50,
        title="Distribution of Anomaly Scores"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with right:
    st.subheader("Steps vs Resting HR")
    df_plot = df.copy()
    df_plot["anomaly_flag"] = df_plot["anomaly_label"].map({1: "Normal", -1: "Anomaly"})

    fig_scatter = px.scatter(
        df_plot,
        x="steps",
        y="resting_hr",
        color="anomaly_flag",
        hover_data=["user_id", "date", "hrv", "sleep_hours", "active_minutes", "anomaly_score"],
        title="Activity vs Resting Heart Rate"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

st.subheader("Most Anomalous Records")
anomalies = df[df["anomaly_label"] == -1].sort_values("anomaly_score", ascending=True)
st.dataframe(anomalies.head(20), use_container_width=True)

st.subheader("Raw Scored Dataset")
st.dataframe(df.head(50), use_container_width=True)
