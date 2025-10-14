import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_loader import load_data

st.title("📈 Analyse des données du churn")

df = chargement_donnée()

if df.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Taux de churn global", f"{df['Churn'].value_counts(normalize=True).get('Yes',0)*100:.1f}%")
col2.metric("Nombre de clients", f"{len(df):,}")
col3.metric("Facture moyenne (€)", f"{df['MonthlyCharges'].mean():.2f}")

st.markdown("---")
st.subheader("Répartition du churn par type de contrat")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Contract", hue="Churn", palette="Set2", ax=ax)
st.pyplot(fig)

st.subheader("Distribution de la facture mensuelle")
fig2, ax2 = plt.subplots()
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, ax=ax2)
st.pyplot(fig2)
