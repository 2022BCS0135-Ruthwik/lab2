import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 6. Print metrics (VERY IMPORTANT)
print(f"MSE: {mse}")
print(f"R2: {r2}")

# 7. Save artifacts
joblib.dump(model, "outputs/model.pkl")

with open("outputs/results.json", "w") as f:
    json.dump({"MSE": mse, "R2": r2}, f, indent=4)
