# --- IMPORTS ---
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="📉 Prédiction de Churn - Telco", page_icon="📉", layout="centered")

# --- TITRE ---
st.title("📉 Prédiction de la Probabilité de Résiliation Client")
st.write("Remplissez les informations du client pour estimer la probabilité qu’il quitte le service (churn).")

# --- CHARGEMENT DU MODELE ---
@st.cache_resource
def load_model():
    with open("modele_telco.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- FORMULAIRE D’ENTRÉE ---
st.subheader("🧩 Données du client")

col1, col2 = st.columns(2)

with col1:
    genre = st.selectbox("Genre", ["Homme", "Femme"])
    client_senior = st.selectbox("Client senior ?", ["Non", "Oui"])
    partenaire = st.selectbox("A un partenaire ?", ["Non", "Oui"])
    personnes_a_charge = st.selectbox("A des personnes à charge ?", ["Non", "Oui"])
    anciennete = st.number_input("Ancienneté (mois)", min_value=0, max_value=120, value=24)
    service_tel = st.selectbox("Service téléphone ?", ["Non", "Oui"])
    lignes_multiples = st.selectbox("Lignes multiples ?", ["Non", "Oui"])

with col2:
    type_internet = st.selectbox("Type d’internet", ["Aucun", "DSL", "Fibre optique"])
    contrat = st.selectbox("Type de contrat", ["Mensuel", "1 an", "2 ans"])
    facturation_elec = st.selectbox("Facturation électronique ?", ["Non", "Oui"])
    methode_paiement = st.selectbox("Méthode de paiement", [
        "Virement automatique", 
        "Carte de crédit automatique", 
        "Chèque électronique", 
        "Chèque postal"
    ])
    facture_mensuelle = st.number_input("Facture mensuelle (€)", min_value=0.0, max_value=200.0, value=50.0)
    facture_totale = st.number_input("Facture totale (€)", min_value=0.0, max_value=10000.0, value=1000.0)

# --- ENCODAGE DES VARIABLES ---
# Même logique que dans ton script de nettoyage

def encode_input():
    data = {
        "Genre": 1 if genre == "Femme" else 0,
        "Client_Senior": 1 if client_senior == "Oui" else 0,
        "Partenaire": 1 if partenaire == "Oui" else 0,
        "Personnes_a_charge": 1 if personnes_a_charge == "Oui" else 0,
        "Anciennete": anciennete,
        "Service_Telephone": 1 if service_tel == "Oui" else 0,
        "Lignes_multiples": 1 if lignes_multiples == "Oui" else 0,
        "Facturation_electronique": 1 if facturation_elec == "Oui" else 0,
        "Facture_mensuelle": facture_mensuelle,
        "Facture_totale": facture_totale,
        "Paiement_virement_auto": 1 if methode_paiement == "Virement automatique" else 0,
        "Paiement_carte_auto": 1 if methode_paiement == "Carte de crédit automatique" else 0,
        "Paiement_cheque_elec": 1 if methode_paiement == "Chèque électronique" else 0,
        "Paiement_cheque_postal": 1 if methode_paiement == "Chèque postal" else 0,
        "DSL": 1 if type_internet == "DSL" else 0,
        "Fibre_internet": 1 if type_internet == "Fibre optique" else 0,
        "Contrat_1_an": 1 if contrat == "1 an" else 0,
        "Contrat_2_ans": 1 if contrat == "2 ans" else 0
    }

    return pd.DataFrame([data])

input_data = encode_input()

# --- PREDICTION ---
if st.button("🔮 Prédire la probabilité de résiliation"):
    try:
        proba = model.predict_proba(input_data)[0][1]
        percent = round(proba * 100, 2)

        st.metric(label="Probabilité de résiliation", value=f"{percent} %")

        # --- Jauge Plotly ---
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percent,
            title={'text': "Probabilité de churn (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if percent > 50 else "green"},
                'steps': [
                    {'range': [0, 25], 'color': "#b6fcd5"},
                    {'range': [25, 50], 'color': "#fff2b0"},
                    {'range': [50, 75], 'color': "#ffd580"},
                    {'range': [75, 100], 'color': "#ff9999"},
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
