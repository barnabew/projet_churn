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


def best_modeles(df):
  log_reg = Pipeline([
      ("scaler", StandardScaler()),
      ("clf", LogisticRegression(max_iter=5000, penalty="l2", solver="lbfgs"))
  ])

  log_reg.fit(X_train, y_train)

  coeffs = log_reg.named_steps["clf"].coef_[0]
  features = X_train.columns
  coef_df = pd.DataFrame({"Feature": features, "Coefficient": coeffs})
  coef_df["Impact"] = np.exp(coeffs)

  coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)

  plt.figure(figsize=(10,6))
  plt.barh(coef_df["Feature"], coef_df["Coefficient"])
  plt.xlabel("Coefficient (poids)")
  plt.ylabel("Variables")
  plt.title("Impact des variables sur le churn (régression logistique)")
  plt.gca().invert_yaxis()
  plt.show()
  
  coef_df.head(10)

  param_grid = {
      "clf__C": [0.01, 0.1, 1, 10], 
      "clf__penalty": ["l1", "l2"],
      "clf__solver": ["liblinear", "saga"] 
  }

  grid = GridSearchCV(log_reg, param_grid, cv=5, scoring="recall", n_jobs=-1)
  grid.fit(X_train, y_train)
  
  print("Meilleurs hyperparamètres:", grid.best_params_)
  print("Meilleur score recall:", grid.best_score_)
  
  best_model = grid.best_estimator_

  y_pred_proba = best_model.predict_proba(X_test)[:,1]

  fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
  roc_auc = roc_auc_score(y_test, y_pred_proba)
  
  plt.figure(figsize=(6,6))
  plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
  plt.plot([0,1],[0,1],'--',color="grey")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate (Recall)")
  plt.title("ROC Curve")
  plt.legend()
  plt.show()

  precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
  pr_auc = auc(recall, precision)
  
  plt.figure(figsize=(6,6))
  plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title("Precision-Recall Curve")
  plt.legend()
  plt.show()

  y_pred = log_reg.predict(X_test)
  y_proba = log_reg.predict_proba(X_test)[:, 1]
  
  print("Matrice de confusion :")
  print(confusion_matrix(y_test, y_pred))
  
  print("\nRapport de classification :")
  print(classification_report(y_test, y_pred))
  
  roc_auc = roc_auc_score(y_test, y_proba)
  print("AUC :", roc_auc)
