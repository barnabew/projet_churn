# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data import chargement_nettoyage
from ml import reg_lineaire, train_test

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="📉 Prédiction de Churn - Telco",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Prédiction de la Résiliation Client (Churn)")
st.write("Entrez les informations principales du client pour estimer le risque de résiliation.")

# --- CHARGEMENT DU MODELE ---
df = chargement_nettoyage()
model = reg_lineaire(df)
X_train, X_test, y_train, y_test = train_test(df)
features = X_train.columns.tolist()

# --- FORMULAIRE SIMPLIFIÉ ---
st.subheader("🧩 Données essentielles du client")

col1, col2 = st.columns(2)

with col1:
    anciennete = st.number_input("Ancienneté (mois)", min_value=0, max_value=120, value=24)
    type_internet = st.selectbox("Type d’internet", ["Aucun", "DSL", "Fibre optique"])

with col2:
    contrat = st.selectbox("Type de contrat", ["Mensuel", "1 an", "2 ans"])
    facture_totale = st.number_input("Total dépensé (€)", min_value=0.0, max_value=10000.0, value=1000.0)

# --- ENCODAGE DES VALEURS UTILISATEUR + MOYENNE POUR LE RESTE ---
def encode_input():
    data_dict = {}
    for col in features:
        # Valeurs des 5 variables essentielles
        if col == "Anciennete":
            data_dict[col] = anciennete
        elif col == "Fibre_internet":
            data_dict[col] = 1 if type_internet == "Fibre optique" else 0
        elif col == "DSL":
            data_dict[col] = 1 if type_internet == "DSL" else 0
        elif col == "Contrat_1_an":
            data_dict[col] = 1 if contrat == "1 an" else 0
        elif col == "Contrat_2_ans":
            data_dict[col] = 1 if contrat == "2 ans" else 0
        elif col == "Facture_totale":
            data_dict[col] = facture_totale
        else:
            # Moyenne de la colonne pour les autres variables
            data_dict[col] = df[col].mean()
    return pd.DataFrame([data_dict])

# --- PRÉDICTION ---
if st.button("🔮 Prédire la probabilité de résiliation"):
    data = encode_input()
    proba = model.predict_proba(data)[0][1]
    percent = round(proba * 100, 2)

    st.metric(label="Probabilité de churn", value=f"{percent} %")

    # Jauge visuelle
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={'text': "Probabilité de churn (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if percent > 50 else "green"}}
    ))
    st.plotly_chart(fig, use_container_width=True)
