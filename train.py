import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE:", mse)
print("R2:", r2)

# Save outputs
os.makedirs("outputs", exist_ok=True)

joblib.dump(model, "outputs/model.pkl")

with open("outputs/results.json", "w") as f:
    json.dump({"MSE": mse, "R2": r2}, f)