# Medoptix Healsight – Hospital Operations Analytics & Forecasting

## 📌 Project Overview

This project analyses hospital operational data (admissions, occupancy, staffing, and flow) to identify trends and build predictive insights for better decision-making.

The goal is to simulate a real-world healthcare analytics pipeline:

* Data ingestion
* SQL database storage
* Data validation
* Feature engineering
* Forecast modelling
* Visualization

---

## 🗂️ Project Structure

```
medoptix-healsight/
│
├── data/raw/                # Raw source CSV files
├── database/               # Database connection logic
├── outputs/figures/        # Saved charts and visuals
├── src/                    # Core pipeline scripts
│   ├── extract_data.py
│   ├── run_sql_checks.py
│   ├── build_dataset.py
│   ├── train_model.py
│
├── medoptix.db             # SQLite database
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Pipeline Stages

### 1. Data Extraction

**File:** `src/extract_data.py`

* Reads raw CSV files
* Loads data into SQLite database

---

### 2. Data Validation (SQL Checks)

**File:** `src/run_sql_checks.py`

* Row counts
* Date ranges
* Duplicate detection
* Table structure validation

---

### 3. Dataset Preparation

**File:** `src/build_dataset.py`

* Converts date columns
* Sorts time series data
* Creates features:

  * Lag features (7, 14 days)
  * Rolling averages
* Generates and saves plots

---

### 4. Model Training

**File:** `src/train_model.py`

* Loads processed dataset
* Trains predictive model
* Outputs evaluation (MAE)

---

## ✅ Current Progress

* [x] Project structure created
* [x] Raw data loaded
* [x] SQLite database connected
* [x] SQL validation checks completed
* [x] Dataset built and cleaned
* [x] Feature engineering implemented
* [x] First visualization created and saved
* [x] First model trained

---

## 🚧 Next Steps (towards 80%)

* [ ] Add multiple visualisations (admissions, overflow, wait times)
* [ ] Generate business insights
* [ ] Identify busiest hospitals/wards
* [ ] Export clean dataset for dashboarding
* [ ] Improve model performance
* [ ] Write full project explanation

---

## 📊 Outputs

* Occupancy trend plot (Hospital 1 – ED Ward)
* Model evaluation (MAE)

---

## 🧠 Skills Demonstrated

* Python (Pandas, NumPy)
* SQL (SQLite)
* Data cleaning & validation
* Feature engineering
* Time-series preparation
* Data visualisation (Matplotlib)
* Predictive modelling

---

## 🎯 Project Goal (Portfolio Angle)

This project demonstrates how raw operational healthcare data can be transformed into:

* actionable insights
* forecasting tools
* decision-support analytics

---

## 🔜 Final Target

A complete analytics project with:

* clean data pipeline
* strong visuals
* business insights
* forecasting model
* dashboard-ready dataset
