import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database.db_source import get_engine

def extract_data():
    print("Starting data extraction...")

    admissions = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "main_admissions.csv"))
    daily_metrics = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "main_daily_metrics.csv"))
    hospitals = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "hospitals.csv"))
    wards = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "wards.csv"))

    print("Extraction complete.")
    
    return {
        "admissions": admissions,
        "daily_metrics": daily_metrics,
        "hospitals": hospitals,
        "wards": wards
    }

def load_to_database(data):
    print("Loading data into database...")

    engine = get_engine()

    for table_name, df in data.items():
        df.to_sql(table_name + "_raw", engine, if_exists="replace", index=False)

    print("Data loaded into database.")

if __name__ == "__main__":
    data = extract_data()

    for name, df in data.items():
        print(f"{name} shape: {df.shape}")

    load_to_database(data)