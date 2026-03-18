import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"


def generate_llm_insight(observations, anomaly):
    if not observations and not anomaly:
        return (
            "Your recent wearable metrics look stable overall. "
            "Your activity, sleep, and recovery-related signals appear generally balanced."
        )

    prompt = f"""
You are a health and wellness AI assistant.

You are given wearable data observations and an anomaly flag.

Your task is to generate a single, short, clear, and supportive health insight.

STRICT RULES:
- Do NOT provide multiple options
- Do NOT ask questions
- Do NOT mention being an AI
- Do NOT diagnose diseases
- Do NOT recommend medications
- Keep it between 2-3 sentences
- Keep the tone calm, supportive, and practical

Observations:
{observations}

Anomaly detected:
{anomaly}

Generate ONLY one final response.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )

    response.raise_for_status()

    result = response.json()
    return result["response"].strip()
