import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data import chargement_nettoyage
from ml import reg_lineaire, train_test
from visuel import ROC,RECALL
st.title("📈 Analyse des données du churn")

@st.cache_resource
def load_model():
    df = chargement_nettoyage()
    model = reg_lineaire(df)
    roc = ROC(model,df)
    recall = RECALL(model,df)
    return model, df, roc, recall  # on retourne df pour calculer les moyennes localement

model, df = load_model()

if df.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Taux de churn global", f"{df["Resiliation"].value_counts(normalize=True).get('Yes',0)*100:.1f}%")
col2.metric("Nombre de clients", f"{len(df):,}")
col3.metric("Facture moyenne (€)", f"{df["Facture_mensuelle"].mean():.2f}")

st.markdown("---")
st.subheader("Répartition du churn par type de contrat")
st.plotly_chart(roc, use_container_width=True)

st.subheader("Distribution de la facture mensuelle")
st.plotly_chart(recall, use_container_width=True)
