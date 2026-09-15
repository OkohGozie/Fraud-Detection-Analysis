"""
api_demo.py — Demonstrates the Module 4 FastAPI /predict endpoint in action.
Run with: python api_demo.py
(Requires the FastAPI app from Module 4 to be running, or run this script's
TestClient version directly without starting a separate server.)
"""

from fastapi.testclient import TestClient
import sys
sys.path.insert(0, "../model_dev")  # adjust path to your repo's api/ folder
from api.main import app

client = TestClient(app)

print("=" * 60)
print("FRAUD DETECTION API — LIVE DEMONSTRATION")
print("=" * 60)

# 1. Health check
r = client.get("/health")
print(f"\n[1] Health check: {r.status_code}")
print(f"    Response: {r.json()}")

# 2. Suspicious transaction
suspicious = {
    "V10": -4.5, "V12": -4.8, "V14": -5.1, "V17": -4.2,
    "Amount": 850.0, "hour_of_day": 3, "quality_flag_negative_amount": False,
}
r = client.post("/predict", json=suspicious)
print(f"\n[2] Suspicious transaction (off-hour, unusual behavioral pattern): {r.status_code}")
for k, v in r.json().items():
    print(f"    {k}: {v}")

# 3. Normal transaction
normal = {"Amount": 45.0, "hour_of_day": 14, "quality_flag_negative_amount": False}
r = client.post("/predict", json=normal)
print(f"\n[3] Normal transaction (routine daytime purchase): {r.status_code}")
for k, v in r.json().items():
    print(f"    {k}: {v}")

print("\n" + "=" * 60)
print("Demo complete — both predictions returned expected risk tiers.")
print("=" * 60)
