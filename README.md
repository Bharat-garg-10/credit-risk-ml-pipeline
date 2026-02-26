# Credit Risk ML Pipeline

This project implements an end-to-end credit risk classification pipeline for **FinTech Solutions Inc.**, a digital lending platform that wants to automate credit risk assessment for personal loans.

## Business Objective

The goal is to build a system that:
- Predicts whether a loan applicant will **default** or **repay**.
- Processes applications in **real-time (<1 second)**.
- Maintains **interpretability** for regulatory compliance.
- Achieves at least **75% recall on default (\"bad\") cases** to minimize credit losses.

## Dataset

- **Name**: German Credit Risk Dataset (or synthetic equivalent)
- **File**: `data/raw/german_credit_data.csv`
- **Rows**: ~1,000 credit applications
- **Target**: `Risk` (`good` / `bad`)

If the original dataset is not available, you can generate a synthetic version by running the provided data generation script (to be added under `src/` or root).

## Project Structure

```text
credit-risk-ml-pipeline/
├── data/
│   ├── raw/
│   │   └── german_credit_data.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Model_Development.ipynb
│   └── 03_Model_Evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── reports/
│   ├── figures/
│   └── technical_report.pdf
│
├── mlruns/
├── requirements.txt
├── README.md
└── .gitignore
```

## Environment Setup (Conda Recommended)

1. Create and activate a Conda environment (Windows / PowerShell or Anaconda Prompt):

```bash
conda create -n credit_risk_env python=3.11
conda activate credit_risk_env
```

2. Install dependencies:

```bash
cd D:\projects\credit-risk-ml-pipeline
pip install -r requirements.txt
```

## How to Run

1. **Generate / place data**
   - Option A: Download `german_credit_data.csv` and place it in `data/raw/`.
   - Option B: Run the synthetic data generator script to create `data/raw/german_credit_data.csv`.

2. **Run notebooks**
   - Start Jupyter:
     ```bash
     jupyter notebook
     ```
   - Open and run, in order:
     - `notebooks/01_EDA_Preprocessing.ipynb`
     - `notebooks/02_Model_Development.ipynb`
     - `notebooks/03_Model_Evaluation.ipynb`

3. **Experiment tracking with MLflow**
   - Ensure MLflow runs are configured in the notebooks or `src/train.py`.
   - To view the UI:
     ```bash
     mlflow ui
     ```

## Results Summary (to be completed)

Document here:
- Final chosen model and key metrics (Accuracy, Precision, Recall, F1, AUC-ROC).
- Recall achieved on `bad` (default) cases.
- Key business interpretation points.

## Contributors

- Your Name (FinTech Solutions ML Engineer)




