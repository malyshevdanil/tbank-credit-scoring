import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Чтобы фронт мог обращаться к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модель и скейлер
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Данные от пользователя


class ClientData(BaseModel):
    age: int
    income: int
    education: int
    work: int
    car: int


@app.post("/score")
def score(data: ClientData):
    X = np.array(
        [[data.age, data.income, data.education, data.work, data.car]])
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0][1]  # вероятность дефолта
    return {"approved": bool(proba < 0.5), "risk": round(float(proba), 2)}
