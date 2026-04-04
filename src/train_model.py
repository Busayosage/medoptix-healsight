import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database.db_source import get_engine

engine = get_engine()

print("Loading dataset...")

query = "SELECT * FROM daily_metrics_raw"
df = pd.read_sql(query, engine)

# Prepare data
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["hospital_id", "ward_code", "date"])

# Feature engineering (same as before)
df["lag_1"] = df.groupby(["hospital_id", "ward_code"])["occupancy"].shift(1)
df["lag_7"] = df.groupby(["hospital_id", "ward_code"])["occupancy"].shift(7)
df["lag_14"] = df.groupby(["hospital_id", "ward_code"])["occupancy"].shift(14)

df["rolling_mean_7"] = df.groupby(["hospital_id", "ward_code"])["occupancy"].transform(lambda x: x.rolling(7).mean())
df["rolling_mean_14"] = df.groupby(["hospital_id", "ward_code"])["occupancy"].transform(lambda x: x.rolling(14).mean())

df = df.dropna()

print("Data ready:", df.shape)

# Define features and target
features = [
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14",
    "admissions", "discharges",
    "staffing_index", "avg_wait_minutes"
]

target = "occupancy"

X = df[features]
y = df[target]

# Train-test split (time-based would be better later, but this is first model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

print("Model trained.")
print("MAE:", mae)