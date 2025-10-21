mport streamlit as st
import pandas as pd
from data import chargement_nettoyage
from ml import reg_lineaire, train_test
from visuel import ROC, RECALL

# --- PAGE CONFIG ---
st.set_page_config(page_title="📊 Performance du Modèle - Churn", page_icon="📊", layout="wide")
st.title("📊 Évaluation du Modèle de Prédiction du Churn")

@st.cache_resource
def load_model():
    df = chargement_nettoyage()
    model = reg_lineaire(df)
    roc_fig = ROC(model, df)
    recall_fig = RECALL(model, df)

    # Calcul des performances principales
    X_train, X_test, y_train, y_test = train_test(df)
    score = model.score(X_test, y_test)  # exactitude
    y_pred = model.predict(X_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, df, roc_fig, recall_fig, score, precision, recall, f1, auc


# --- CHARGEMENT DU MODÈLE ---
model, df, roc_fig, recall_fig, score, precision, recall, f1, auc = load_model()

if df.empty:
    st.warning("⚠️ Aucune donnée disponible.")
    st.stop()

# --- KPI : METRIQUES DE PERFORMANCE ---
st.subheader("📈 Indicateurs de performance du modèle")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Exactitude (Accuracy)", f"{score:.2%}")
col2.metric("Précision", f"{precision:.2%}")
col3.metric("Rappel (Recall)", f"{recall:.2%}")
col4.metric("F1-Score", f"{f1:.2%}")

st.metric("AUC (Air sous la courbe ROC)", f"{auc:.2f}")

st.markdown("---")

# --- COURBES ---
st.subheader("🎯 Courbe ROC (Receiver Operating Characteristic)")
st.plotly_chart(roc_fig, use_container_width=True)
st.markdown("""
> 📘 **Interprétation :**  
> La courbe ROC illustre la capacité du modèle à distinguer les clients susceptibles de résilier.  
> Une AUC proche de **1.0** indique une très bonne performance du modèle, tandis qu’une AUC de **0.5** correspond à un modèle aléatoire.
""")

st.markdown("---")

st.subheader("📊 Courbe Précision - Rappel")
st.plotly_chart(recall_fig, use_container_width=True)
st.markdown("""
> 📘 **Interprétation :**  
> Cette courbe évalue le compromis entre **précision** (éviter les faux positifs) et **rappel** (identifier tous les vrais churners).  
> Plus la courbe est proche du coin supérieur droit, plus le modèle est performant pour détecter les clients à risque de résiliation.
""")

st.markdown("---")

# --- CONCLUSION ---
st.subheader("🧠 Choix du modèle")
st.write("""
Nous avons choisi d’utiliser une **régression logistique** comme modèle de base.  
Ce modèle présente plusieurs avantages :
- Simplicité d’interprétation des coefficients ;
- Rapidité d’entraînement ;
- Bon équilibre entre biais et variance sur ce type de données.

Les hyperparamètres ont été sélectionnés après plusieurs essais afin d’obtenir le meilleur compromis entre **rappel** et **précision**,  
ce qui est essentiel pour identifier efficacement les clients susceptibles de résilier.
""")
