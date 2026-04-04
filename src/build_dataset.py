import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt

# Connect to database
conn = sqlite3.connect("medoptix.db")

# Load data
query = "SELECT * FROM daily_metrics_raw"
df = pd.read_sql(query, conn)

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Feature engineering
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Example aggregation (keep your existing logic if different)
final_df = df.copy()

# Ensure processed data folder exists
os.makedirs("data/processed", exist_ok=True)

# ✅ UPDATED OUTPUT PATH
output_path = "data/processed/final_dataset.csv"

# Save dataset
final_df.to_csv(output_path, index=False)

print(f"Final dataset saved to {output_path}")