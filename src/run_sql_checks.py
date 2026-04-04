import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database.db_source import get_engine

engine = get_engine()

print("\n--- Row Counts ---")

queries = {
    "admissions": "SELECT COUNT(*) as count FROM admissions_raw",
    "daily_metrics": "SELECT COUNT(*) as count FROM daily_metrics_raw",
    "hospitals": "SELECT COUNT(*) as count FROM hospitals_raw",
    "wards": "SELECT COUNT(*) as count FROM wards_raw"
}

for name, query in queries.items():
    df = pd.read_sql(query, engine)
    print(f"{name}: {df['count'][0]}")

print("\n--- Date Range ---")

date_query = """
SELECT MIN(date) as min_date, MAX(date) as max_date 
FROM daily_metrics_raw
"""

df_dates = pd.read_sql(date_query, engine)
print(df_dates)

print("\n--- Duplicate Check ---")

dup_query = """
SELECT date, hospital_id, ward_code, COUNT(*) as count
FROM daily_metrics_raw
GROUP BY date, hospital_id, ward_code
HAVING COUNT(*) > 1
"""

df_dup = pd.read_sql(dup_query, engine)

if df_dup.empty:
    print("No duplicates found")
else:
    print("Duplicates found")
    print(df_dup.head())

print("\n--- Table Columns ---")

tables = ["admissions_raw", "daily_metrics_raw", "hospitals_raw", "wards_raw"]

for table in tables:
    print(f"\n{table} columns:")
    query = f"PRAGMA table_info({table});"
    df = pd.read_sql(query, engine)
    print(df[["name", "type"]])