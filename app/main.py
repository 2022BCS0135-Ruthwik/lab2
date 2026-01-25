from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(
    fixed_acidity: float,
    volatile_acidity: float,
    citric_acid: float,
    residual_sugar: float,
    chlorides: float,
    free_sulfur_dioxide: float,
    total_sulfur_dioxide: float,
    density: float,
    pH: float,
    sulphates: float,
    alcohol: float
):
    X = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                   residual_sugar, chlorides, free_sulfur_dioxide,
                   total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    pred = model.predict(X)[0]

    return {
        "name": "Ruthwik",
        "roll_no": "2022BCS0135",
        "wine_quality": round(float(pred), 2)
    }
