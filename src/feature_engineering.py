"""
Feature engineering utilities for the German Credit Risk dataset.

Implements at least three domain-inspired features, e.g.:
- Credit_to_Duration_Ratio
- Age_Group
- High_Risk_Purpose
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def add_credit_to_duration_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add Credit_to_Duration_Ratio = Credit amount / Duration."""
    df = df.copy()
    df["Credit_to_Duration_Ratio"] = df["Credit amount"] / df["Duration"].clip(lower=1)
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket Age into categorical age groups."""
    df = df.copy()
    bins = [18, 25, 35, 50, 120]
    labels = ["18-25", "26-35", "36-50", "50+"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True, include_lowest=True)
    return df


def add_high_risk_purpose(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary High_Risk_Purpose flag.

    You can adjust the set of high-risk purposes based on domain insights.
    """
    df = df.copy()
    high_risk_purposes = {"vacation/others", "repairs", "business"}
    df["High_Risk_Purpose"] = df["Purpose"].isin(high_risk_purposes).astype(int)
    return df


def apply_all_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    df_fe = add_credit_to_duration_ratio(df)
    df_fe = add_age_group(df_fe)
    df_fe = add_high_risk_purpose(df_fe)
    return df_fe

def evaluate_model(y_true, y_pred, y_proba):
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred)
    metrics["Recall"] = recall_score(y_true, y_pred)
    metrics["F1-Score"] = f1_score(y_true, y_pred)
    metrics["AUC-ROC"] = roc_auc_score(y_true, y_proba)
    return metrics

def log_model_to_mlflow(model, model_name, X_test, y_test, y_pred, y_proba):
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    with mlflow.start_run(run_name=model_name):
        
        # Log Parameters
        mlflow.log_param("model_type", model_name)
        
        if hasattr(model, "get_params"):
            for param, value in model.get_params().items():
                mlflow.log_param(param, value)
        
        # Log Metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("auc_roc", roc_auc_score(y_test, y_proba))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        
        # Log Model
        mlflow.sklearn.log_model(model, "model")



