import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def modeles(df):
  X = df.drop(columns=["ID_Client", "Resiliation"])
  y = df["Resiliation"]


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  

  models = {
      "Logistic Regression": LogisticRegression(max_iter=5000, class_weight='balanced'),
      "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
      "XGBoost": XGBClassifier(eval_metric='logloss'),
      "LightGBM": LGBMClassifier()
  }
  

  results = {}
  

  for name, model in models.items():
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      y_proba = model.predict_proba(X_test)[:,1]
      
      report = classification_report(y_test, y_pred, output_dict=True)
      Auc = roc_auc_score(y_test, y_proba)
      
      results[name] = {
          "Accuracy": report["accuracy"],
          "Recall_1": report["1"]["recall"],
          "Precision_1": report["1"]["precision"],
          "F1_1": report["1"]["f1-score"],
          "AUC": Auc
      }
  

  results_df = pd.DataFrame(results).T.sort_values(by="AUC", ascending=False)
  print(results_df)
