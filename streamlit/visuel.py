import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def poids(df):
  return px.bar(df, x="Feature", y="Coefficient", title="Impact des variables sur le churn (régression logistique)")

def ROC(model):
  # Création de la figure
  fig = go.Figure()
  
  # Ajouter la courbe ROC
  fig.add_trace(go.Scatter(
      x=fpr,
      y=tpr,
      mode='lines',
      name=f"AUC = {roc_auc:.3f}",
      line=dict(color='blue', width=2)
  ))
  
  # Ajouter la diagonale (cas aléatoire)
  fig.add_trace(go.Scatter(
      x=[0,1],
      y=[0,1],
      mode='lines',
      line=dict(color='grey', dash='dash'),
      showlegend=False
  ))
  
  # Mettre les labels et titre
  fig.update_layout(
      title="ROC Curve",
      xaxis_title="False Positive Rate",
      yaxis_title="True Positive Rate (Recall)",
      width=600,
      height=600
  )
  
  # Afficher le graphique interactif
  return fig
