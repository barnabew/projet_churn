import streamlit as st
from ml import reg_lineaire, train_test
from data import chargement_nettoyage
import pickle

df=chargement_nettoyage()

model_data = {
    "model": reg_lineaire(df),
    "features": train_test(df)[0].columns.tolist()
}

with open("modele.pkl", "wb") as f:
    pickle.dump(model_data, f)

st.set_page_config(
    page_title="Analyse et Prédiction du Churn Client",
    page_icon="📉",
    layout="wide"
)

st.title("📊 Projet : Prédiction du Churn Client Télécom")
st.write("Bienvenue dans l’application d’analyse et de prédiction du churn.")

st.sidebar.success("Choisissez une page ci-dessus pour commencer.")
