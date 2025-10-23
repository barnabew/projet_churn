# Projet : Prédiction du churn des clients télécoms

## Contexte

Dans un marché des télécommunications très concurrentiel, la fidélisation des clients représente un enjeu stratégique majeur.  
Acquérir un nouveau client coûte souvent beaucoup plus cher que de conserver un abonné existant.  
Anticiper les résiliations — ou **churn** — permet ainsi d’optimiser les campagnes marketing et d’améliorer la satisfaction client.

Ce projet vise à **prédire la résiliation des clients** d’une entreprise de télécommunications à partir du jeu de données public *Telco Customer Churn*.  
L’objectif est double :
- Identifier les facteurs qui influencent le plus la décision de résiliation ;
- Développer un modèle de machine learning capable d’anticiper les clients à risque.

Les résultats obtenus doivent permettre d’appuyer les décisions stratégiques en matière de **fidélisation** et d’**optimisation des offres**.

---

## Analyses réalisées

L’étude a été menée en plusieurs étapes, depuis la préparation des données jusqu’à l’évaluation des performances prédictives.

L’exploration initiale a permis d’identifier les variables les plus pertinentes (type de contrat, ancienneté, facture mensuelle, type d’accès Internet, etc.) et de nettoyer les données.  
Les valeurs manquantes ont été traitées, les colonnes textuelles harmonisées et les variables catégorielles transformées en variables numériques à l’aide de la fonction `get_dummies`.

Plusieurs modèles ont ensuite été testés et comparés :
- Régression logistique  
- Forêt aléatoire (*Random Forest*)  
- XGBoost  
- LightGBM  

L’évaluation s’est appuyée sur plusieurs métriques : **rappel (recall)**, **précision**, **F1-score** et **AUC**.  
La **régression logistique** a été retenue comme modèle principal pour sa bonne performance en rappel, sa stabilité et sa facilité d’interprétation.  
Une optimisation des hyperparamètres a été réalisée via `GridSearchCV` pour ajuster la régularisation et le solveur.

Les performances finales ont été mesurées à l’aide des courbes **ROC** et **Precision–Recall**, de la matrice de confusion et du rapport de classification.  
L’analyse des coefficients de la régression logistique a ensuite permis d’interpréter l’impact de chaque variable sur la probabilité de résiliation.

---

## Résultats clés

Le modèle final présente un **AUC de 0.86** et un **rappel supérieur à 80 %**, ce qui permet de détecter efficacement la majorité des clients susceptibles de résilier.  
L’analyse des coefficients met en évidence plusieurs facteurs majeurs :

- Le **type de contrat** : les contrats longue durée sont fortement associés à une baisse du risque de churn.  
- L’**ancienneté** : les nouveaux clients présentent une probabilité plus élevée de résiliation.  
- La **facture mensuelle** : un montant plus élevé est corrélé à une plus forte propension au churn.  
- Le **type d’accès Internet** influence également la fidélité, certaines technologies étant plus associées au départ des clients.

Ces résultats ont ensuite permis de **construire un modèle prédictif robuste**,  
capable d’estimer la probabilité de résiliation pour chaque client à partir de ses caractéristiques.  
Ce modèle sert de base à l’interface **Streamlit** développée pour visualiser les performances et effectuer des **prédictions interactives** en temps réel.


---

## Organisation du projet

Le notebook [`Projet_churn.ipynb`](https://github.com/barnabew/projet_churn/blob/main/Projet_churn.ipynb) contient toutes les explications détaillées sur le **traitement des données**,  
le **nettoyage**, le **test des différents modèles** et l’**analyse des performances**.  
Il constitue la base exploratoire du projet, permettant de documenter chaque étape du raisonnement.

Le dossier [`streamlit/`](https://github.com/barnabew/projet_churn/tree/main/streamlit) reprend le même code,  
mais il a été **structuré en plusieurs fichiers** afin de rendre l’application plus **lisible**, **modulaire** et **facile à maintenir**.  
Cette séparation du code (data, machine learning, visualisation, interface) permet une meilleure réutilisation et simplifie les futures évolutions du projet.  

---

## Application Streamlit

Une **interface Streamlit** a été développée afin de permettre une utilisation interactive du modèle.  
Elle se compose de plusieurs pages :

1. **Introduction** – présentation du contexte, des objectifs et de la démarche analytique.  
2. **Performance du modèle** – évaluation des résultats obtenus avec les principales métriques (AUC, rappel, précision, F1-score) et visualisation des courbes ROC et Précision–Rappel.  
3. **Prédiction interactive** – simulation en temps réel de la probabilité de churn à partir des caractéristiques d’un client.

L’application est disponible :  [Accéder à l'application](https://projetchurn.streamlit.app/).

---

## Résultats et recommandations

L’analyse met en évidence que la résiliation des clients est influencée par plusieurs variables contractuelles et comportementales.  
Les recommandations issues de ce travail sont les suivantes :

1. **Fidéliser les nouveaux clients** par des offres promotionnelles ou des réductions ciblées après les premiers mois d’abonnement.  
2. **Encourager la souscription à des contrats longue durée** pour réduire le taux de churn.  
3. **Surveiller les clients à forte facturation** et leur proposer des avantages personnalisés.  
4. **Améliorer la qualité du service client**, en particulier pour les nouveaux abonnés, plus susceptibles de résilier.

---

## Pistes d’amélioration

- Intégrer des méthodes d’équilibrage de classes (par ex. **SMOTE**) afin de mieux gérer le déséquilibre entre churners et non-churners.  
- Expérimenter d’autres modèles (par ex. **CatBoost**) pour affiner la comparaison.  
- Déployer le modèle via une API pour faciliter son intégration dans les outils internes de gestion client.

---

## Environnement

- **Langage** : Python 3.10  
- **Principales librairies** : pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, plotly, streamlit  
- **Visualisation** : Matplotlib / Plotly  
- **Interface** : Streamlit

---

## Auteur

Projet réalisé par **Barnabé W.** dans le cadre d’un travail d’analyse et de modélisation du churn client.
