# --- PAGE CONFIG ---
st.set_page_config(page_title="Prédiction de Churn", page_icon="📉", layout="centered")

# --- TITRE ---
st.title("📉 Prédiction de la Probabilité de Churn")
st.write("Entrez les informations du client pour estimer la probabilité qu’il quitte le service.")

# --- CHARGEMENT DU MODELE ---
@st.cache_resource
def load_model():
    with open("modele.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- FORMULAIRE D’ENTRÉE ---
st.subheader("🧩 Données du client")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Âge", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Ancienneté (mois)", min_value=0, max_value=120, value=24)
    monthly_charges = st.number_input("Facture mensuelle (€)", min_value=0.0, max_value=200.0, value=50.0)

with col2:
    total_charges = st.number_input("Total dépensé (€)", min_value=0.0, max_value=5000.0, value=1000.0)
    support_calls = st.number_input("Nombre d’appels au support", min_value=0, max_value=50, value=3)
    has_contract = st.selectbox("Type de contrat", ["Mensuel", "Annuel"])

# --- ENCODAGE DES VARIABLES CATEGORIELLES ---
contract_map = {"Mensuel": 0, "Annuel": 1}

input_data = pd.DataFrame({
    "age": [age],
    "tenure": [tenure],
    "monthly_charges": [monthly_charges],
    "total_charges": [total_charges],
    "support_calls": [support_calls],
    "has_contract": [contract_map[has_contract]]
})

# --- BOUTON DE PREDICTION ---
if st.button("🔮 Prédire la probabilité de churn"):
    prediction = model.predict_proba(input_data)[0][1]  # Probabilité de churn
    percent = round(prediction * 100, 2)

    st.metric(label="Probabilité de churn", value=f"{percent} %")

    # Optionnel : jauge visuelle
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={'text': "Probabilité de churn (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if percent > 50 else "green"}}
    ))
    st.plotly_chart(fig, use_container_width=True)
