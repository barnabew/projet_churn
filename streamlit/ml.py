import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def train_test(df):
  X = df.drop(columns=["ID_Client", "Resiliation"])
  y = df["Resiliation"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test


def reg_lineaire(df):
  X = df.drop(columns=["ID_Client", "Resiliation"])
  y = df["Resiliation"]
  
  log_reg = Pipeline([
      ("scaler", StandardScaler()),
      ("clf", LogisticRegression(max_iter=5000, penalty="l2", solver="lbfgs"))
  ])
  
  param_grid = {
      "clf__C": [0.01, 0.1, 1, 10], 
      "clf__penalty": ["l1", "l2"],
      "clf__solver": ["liblinear", "saga"] 
  }

  grid = GridSearchCV(log_reg, param_grid, cv=5, scoring="recall", n_jobs=-1)
  grid.fit(X_train, y_train)
  
  best_model = grid.best_estimator_

  return best_model


def coefficient(df):
  X = df.drop(columns=["ID_Client", "Resiliation"])
  y = df["Resiliation"]

  log_reg = Pipeline([
      ("scaler", StandardScaler()),
      ("clf", LogisticRegression(max_iter=5000, penalty="l2", solver="lbfgs"))
  ])

  coeffs = log_reg.named_steps["clf"].coef_[0]
  features = X_train.columns
  coef_df = pd.DataFrame({"Feature": features, "Coefficient": coeffs})
  coef_df["Impact"] = np.exp(coeffs)

  coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)

  return coef_df
