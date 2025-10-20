# --- IMPORTS ---
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="📉 Prédiction de Churn - Telco", page_icon="📉", layout="centered")

st.title("📉 Prédiction de la Résiliation Client (Churn)")
st.write("Entrez les informations principales du client pour estimer le risque de résiliation.")

# --- CHARGEMENT DU MODELE ---
@st.cache_resource
def load_model():
    with open("modele.pkl", "wb") as f:
        return pickle.load(f)

model = load_model()

# --- FORMULAIRE ---
st.subheader("🧩 Données du client")

col1, col2 = st.columns(2)

with col1:
    anciennete = st.number_input("Ancienneté (mois)", min_value=0, max_value=120, value=24)
    type_internet = st.selectbox("Type d’internet", ["Aucun", "DSL", "Fibre optique"])
    securite_en_ligne = st.selectbox("Sécurité en ligne ?", ["Non", "Oui"])
    streaming_films = st.selectbox("Regarde des films en streaming ?", ["Non", "Oui"])

with col2:
    contrat = st.selectbox("Type de contrat", ["Mensuel", "1 an", "2 ans"])
    facture_mensuelle = st.number_input("Facture mensuelle (€)", min_value=0.0, max_value=200.0, value=50.0)
    facture_totale = st.number_input("Facture totale (€)", min_value=0.0, max_value=10000.0, value=1000.0)
    facturation_elec = st.selectbox("Facturation électronique ?", ["Non", "Oui"])

# --- ENCODAGE ---
def encode_input():
    data = {
        "Anciennete": anciennete,
        "Fibre_internet": 1 if type_internet == "Fibre optique" else 0,
        "DSL": 1 if type_internet == "DSL" else 0,
        "Facture_mensuelle": facture_mensuelle,
        "Facture_totale": facture_totale,
        "Contrat_1_an": 1 if contrat == "1 an" else 0,
        "Contrat_2_ans": 1 if contrat == "2 ans" else 0,
        "Securite_en_ligne": 1 if securite_en_ligne == "Oui" else 0,
        "Streaming_Films": 1 if streaming_films == "Oui" else 0,
        "Facturation_electronique": 1 if facturation_elec == "Oui" else 0
    }
    return pd.DataFrame([data])

input_data = encode_input()

# --- PREDICTION ---
if st.button("🔮 Prédire la probabilité de résiliation"):
    proba = model.predict_proba(input_data)[0][1]
    percent = round(proba * 100, 2)

    st.metric(label="Probabilité de churn", value=f"{percent} %")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={'text': "Probabilité de churn (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if percent > 50 else "green"}}
    ))
    st.plotly_chart(fig, use_container_width=True)
