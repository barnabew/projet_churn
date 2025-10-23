import streamlit as st
import pandas as pd
from data import chargement_nettoyage
from ml import reg_lineaire, train_test
from visuel import ROC, RECALL

# --- PAGE CONFIG ---
st.set_page_config(page_title=" Performance du Modèle - Churn", layout="wide")
st.title(" Performance du modèle de Prédiction du Churn")

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
    st.warning(" Aucune donnée disponible.")
    st.stop()

# --- KPI : METRIQUES DE PERFORMANCE ---
st.subheader(" Indicateurs de performance du modèle")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{score:.2%}")
col2.metric("Précision", f"{precision:.2%}")
col3.metric("Rappel", f"{recall:.2%}")
col4.metric("F1-score", f"{f1:.2%}")
col5.metric("AUC", f"{auc:.2f}")

with st.expander("ℹ️ Voir les explications des indicateurs"):
    st.markdown("""
    - **Accuracy** : proportion totale de prédictions correctes.  
    - **Précision** : parmi les churn prédits, combien sont réels.  
    - **Rappel** : parmi les vrais churn, combien ont été détectés.  
    - **F1-score** : équilibre entre précision et rappel.  
    - **AUC (ROC)** : capacité du modèle à distinguer churners et non-churners.  
    """)

st.markdown("---")



# --- CONCLUSION ---
st.subheader("Analyse technique des performances du modèle")
st.write("""
Le modèle de régression logistique présente une **AUC de 0.86**, illustrant une bonne capacité à discriminer les clients à risque de résiliation 
des clients fidèles. Cette performance témoigne d’un modèle stable, à la fois robuste et interprétable.

Les métriques principales montrent une **précision d’environ 0.80** et un **rappel de 0.72**, indiquant un compromis satisfaisant 
entre la qualité des prédictions positives et la capacité à détecter la majorité des churners. 
Le **F1-score**, proche de 0.76, confirme cet équilibre global entre précision et rappel.  
Le modèle maintient par ailleurs une **bonne cohérence entre les ensembles d’entraînement et de test**, 
ce qui démontre une généralisation correcte sans surapprentissage notable.

Les analyses complémentaires, présentées dans le fichier **`Projet_churn.ipynb`** disponible sur le dépôt GitHub, 
montrent qu’un travail approfondi d’ajustement des **hyperparamètres** (pénalisation, régularisation, solver) a été réalisé, 
ainsi qu’une **évaluation comparative** avec d’autres modèles tels que **Random Forest** et **XGBoost**.  
Ces tests ont confirmé que la régression logistique offre le **meilleur compromis entre performance, interprétabilité et simplicité de déploiement**.

Enfin, une étude du **seuil de décision** et de la **pondération des classes** a permis d’améliorer la détection des clients churners 
sans détériorer excessivement la précision globale.  
Les résultats finaux montrent un modèle bien calibré, capable d’identifier efficacement les clients à risque tout en conservant une cohérence métier.

Ainsi, la régression logistique constitue ici un **modèle de référence** fiable et opérationnel, 
dont le développement complet et les expérimentations sont détaillés dans le notebook d’analyse joint au projet.
""")

st.markdown("---")

# --- COURBES ---
st.subheader(" Courbe ROC (Receiver Operating Characteristic)")
st.plotly_chart(roc_fig, use_container_width=True)
with st.expander(" Interprétation"):
    st.markdown("""
    La courbe ROC illustre la capacité du modèle à distinguer les clients susceptibles de résilier de ceux qui restent fidèles. 
    L’AUC obtenue est de 0.86, ce qui indique une bonne performance. 
    Une valeur proche de 1 traduit un modèle performant, tandis qu’une valeur de 0.5 correspond à un modèle aléatoire. 
    Ainsi, avec un AUC de 0.86, le modèle identifie correctement les clients churners dans la grande majorité des cas.
    """)

st.markdown("---")

st.subheader(" Courbe Précision - Rappel")
st.plotly_chart(recall_fig, use_container_width=True)
with st.expander(" Interprétation"):
    st.markdown("""
    La courbe Precision–Recall permet d’évaluer plus finement la qualité des prédictions sur la classe minoritaire, ici la résiliation. 
    On observe une bonne précision (environ 0.7 à 0.8) pour des valeurs de rappel intermédiaires, 
    ce qui signifie que le modèle détecte correctement une part importante des clients à risque tout en limitant le nombre de fausses alertes. 
    Comme souvent, lorsque le rappel augmente, la précision diminue, illustrant le compromis classique entre ces deux indicateurs.
    """)


