from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

HYPER = DATA / "hyperspectral_data.csv"
MOISTURE = DATA / "soil_moisture_data.csv"

MODEL = MODELS / "disease_risk_model.joblib"