import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data import chargement_nettoyage
from ml import reg_lineaire

st.title("📈 Analyse des données du churn")

df = chargement_nettoyage()

if df.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Taux de churn global", f"{df['Churn'].value_counts(normalize=True).get('Yes',0)*100:.1f}%")
col2.metric("Nombre de clients", f"{len(df):,}")
col3.metric("Facture moyenne (€)", f"{df['MonthlyCharges'].mean():.2f}")

st.markdown("---")
st.subheader("Répartition du churn par type de contrat")
st.plotly_chart(ROC(reg_lineaire(df),df), use_container_width=True)

st.subheader("Distribution de la facture mensuelle")
st.plotly_chart(RECALL(reg_lineaire(df),df), use_container_width=True)
