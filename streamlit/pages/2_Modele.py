# --- IMPORTS ---
import streamlit as st
import pandas as pd
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

# --- CHARGEMENT DU MODELE (une seule fois) ---
@st.cache_resource
def load_model():
    df = chargement_nettoyage()
    model = reg_lineaire(df)
    X_train, X_test, y_train, y_test = train_test(df)
    features = X_train.columns.tolist()
    data_mean = df.mean()
    return model, features, df, data_mean

model, features, df, data_mean = load_model()

# --- FORMULAIRE SIMPLIFIÉ ---
st.subheader("🧩 Données essentielles du client")

col1, col2 = st.columns(2)

with col1:
    anciennete = st.number_input("Ancienneté (mois)", min_value=0, max_value=120, value=24)
    type_internet = st.selectbox("Type d’internet", ["Aucun", "DSL", "Fibre optique"])
    services = st.multiselect(
        "Services supplémentaires",
        ["Streaming Films", "Streaming TV", "Sécurité en ligne"]
    )

with col2:
    contrat = st.selectbox("Type de contrat", ["Mensuel", "1 an", "2 ans"])
    facture_totale = st.number_input("Total dépensé (€)", min_value=0.0, max_value=10000.0, value=1000.0)

# --- ENCODAGE ---
def encode_input():
    data = data_mean.copy()  # moyenne pour toutes les autres variables
    data["Anciennete"] = anciennete
    data["Fibre_internet"] = 1 if type_internet == "Fibre optique" else 0
    data["DSL"] = 1 if type_internet == "DSL" else 0
    data["Contrat_1_an"] = 1 if contrat == "1 an" else 0
    data["Contrat_2_ans"] = 1 if contrat == "2 ans" else 0
    data["Facture_totale"] = facture_totale
    data["Streaming_Films"] = 1 if "Streaming Films" in services else 0
    data["Streaming_TV"] = 1 if "Streaming TV" in services else 0
    data["Securite_en_ligne"] = 1 if "Sécurité en ligne" in services else 0
    return pd.DataFrame([data])

# --- PRÉDICTION ---
if st.button("🔮 Prédire la probabilité de résiliation"):
    input_data = encode_input()
    proba = model.predict_proba(input_data)[0][1]
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
