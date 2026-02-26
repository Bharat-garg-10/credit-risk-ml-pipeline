"""
Utility script to generate a synthetic German credit dataset, based on the
problem statement specification, and save it to `data/raw/german_credit_data.csv`.
"""

import numpy as np
import pandas as pd


def generate_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    data = {
        "Age": np.random.randint(18, 75, n_samples),
        "Sex": np.random.choice(["male", "female"], n_samples),
        "Job": np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        "Housing": np.random.choice(["own", "rent", "free"], n_samples, p=[0.6, 0.3, 0.1]),
        "Saving accounts": np.random.choice(
            ["little", "moderate", "quite rich", "rich", np.nan],
            n_samples,
            p=[0.4, 0.3, 0.1, 0.1, 0.1],
        ),
        "Checking account": np.random.choice(
            ["little", "moderate", "rich", np.nan],
            n_samples,
            p=[0.5, 0.25, 0.15, 0.1],
        ),
        "Credit amount": np.random.randint(250, 20000, n_samples),
        "Duration": np.random.randint(4, 72, n_samples),
        "Purpose": np.random.choice(
            [
                "car",
                "furniture",
                "radio/TV",
                "education",
                "business",
                "domestic appliances",
                "repairs",
                "vacation/others",
            ],
            n_samples,
            p=[0.3, 0.15, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05],
        ),
    }

    def generate_risk(row: pd.Series) -> str:
        risk_score = 0

        # Age factor
        if row["Age"] < 25:
            risk_score += 15
        elif row["Age"] > 60:
            risk_score += 10

        # Credit amount and duration factor
        if row["Credit amount"] > 10000:
            risk_score += 20
        if row["Duration"] > 36:
            risk_score += 15

        # Savings and checking
        if pd.isna(row["Saving accounts"]) or row["Saving accounts"] == "little":
            risk_score += 20
        if pd.isna(row["Checking account"]) or row["Checking account"] == "little":
            risk_score += 15

        # Job stability
        if row["Job"] == 0:
            risk_score += 25

        # Random noise
        risk_score += np.random.randint(-10, 10)

        # Threshold
        return "bad" if risk_score > 50 else "good"

    df = pd.DataFrame(data)
    df["Risk"] = df.apply(generate_risk, axis=1)
    return df


def main() -> None:
    df = generate_dataset()
    output_path = "data/raw/german_credit_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset created at: {output_path}")
    print(f"Rows: {len(df)}")
    print("Risk distribution:")
    print(df["Risk"].value_counts())


if __name__ == "__main__":
    main()


