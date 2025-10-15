import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from ml import reg_lineaire, train_test
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import plotly.graph_objects as go

def poids(df):
  return px.bar(df, x="Feature", y="Coefficient", title="Impact des variables sur le churn (régression logistique)")

def ROC(model,df):

  X_train, X_test, y_train, y_test = train_test(df)
  y_pred_proba = reg_lineaire(df).predict_proba(X_test)[:,1]
  
  fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
  roc_auc = roc_auc_score(y_test, y_pred_proba)
  
  fig = go.Figure()
  
  
  fig.add_trace(go.Scatter(
      x=fpr,
      y=tpr,
      mode='lines',
      name=f"AUC = {roc_auc:.3f}",
      line=dict(color='blue', width=2)
  ))
  
  
  fig.add_trace(go.Scatter(
      x=[0,1],
      y=[0,1],
      mode='lines',
      line=dict(color='grey', dash='dash'),
      showlegend=False
  ))
  
 
  fig.update_layout(
      title="ROC Curve",
      xaxis_title="False Positive Rate",
      yaxis_title="True Positive Rate (Recall)",
      width=600,
      height=600
  )
  
 
  return fig


def RECALL(model,df):
  X_train, X_test, y_train, y_test = train_test(df)
  y_pred_proba = reg_lineaire(df).predict_proba(X_test)[:,1]
  
  precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
  pr_auc = auc(recall, precision)

  # Création de la figure
  fig = go.Figure()
  
  # Ajouter la courbe Precision-Recall
  fig.add_trace(go.Scatter(
      x=recall,
      y=precision,
      mode='lines',
      name=f"AUC = {pr_auc:.3f}",
      line=dict(color='green', width=2)
  ))
  
  # Mettre les labels et titre
  fig.update_layout(
      title="Precision-Recall Curve",
      xaxis_title="Recall",
      yaxis_title="Precision",
      width=600,
      height=600
  )
  
  # Afficher le graphique interactif
  return fig
