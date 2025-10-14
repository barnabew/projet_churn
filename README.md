# Projet : Prédiction du churn des clients télécoms

## Contexte

Ce projet vise à prédire la résiliation des clients (churn) d’une entreprise de télécommunications à partir du jeu de données public *Telco Customer Churn*.
L’objectif est d’identifier les facteurs les plus déterminants dans la décision de résiliation et de développer un modèle de machine learning capable d’anticiper les clients à risque.
Les résultats obtenus doivent permettre d’appuyer les décisions stratégiques en matière de fidélisation et d’optimisation des offres.

## Analyses réalisées

L’étude a été menée en plusieurs étapes, depuis la préparation des données jusqu’à l’évaluation des performances prédictives.

L’exploration initiale a permis d’identifier les variables les plus pertinentes (type de contrat, ancienneté, facture mensuelle, type d’accès Internet, etc.) et de nettoyer les données. Les valeurs manquantes ont été traitées, les colonnes textuelles harmonisées et les variables catégorielles transformées en variables numériques à l’aide de la fonction `get_dummies`.

Plusieurs modèles ont ensuite été testés et comparés :
- Régression logistique
- Forêt aléatoire (*Random Forest*)
- XGBoost
- LightGBM

L’évaluation s’est appuyée sur plusieurs métriques : **rappel (recall)**, **précision**, **F1-score** et **AUC**.
La régression logistique a été retenue comme modèle principal en raison de sa bonne performance en rappel et de sa lisibilité. Une optimisation des hyperparamètres a été réalisée à l’aide de `GridSearchCV` pour ajuster la régularisation et le solveur.

Les performances finales ont été mesurées à l’aide des courbes **ROC** et **Precision-Recall**, de la matrice de confusion et du rapport de classification. L’analyse des coefficients de la régression logistique a permis d’interpréter l’impact de chaque variable sur la probabilité de résiliation.

## Résultats clés

Le modèle final présente un **rappel supérieur à 80 %**, ce qui permet de détecter efficacement la majorité des clients susceptibles de résilier.
L’analyse des coefficients met en évidence plusieurs facteurs majeurs :
- Le **type de contrat** : les contrats longue durée sont fortement associés à une baisse du risque de churn.
- L’**ancienneté** : les nouveaux clients présentent une probabilité plus élevée de résiliation.
- La **facture mensuelle** : un montant plus élevé est corrélé à une plus forte propension au churn.
- Le **type d’accès Internet** influence également la fidélité, certaines technologies étant plus associées au départ des clients.

## Organisation du projet

Le notebook `Projet_churn.ipynb` comprend l’ensemble des traitements et analyses :
- Préparation et nettoyage des données
- Entraînement et comparaison des modèles
- Optimisation et évaluation des performances
- Interprétation des résultats et visualisations associées

Les librairies principales utilisées sont : **pandas**, **numpy**, **scikit-learn**, **xgboost**, **lightgbm** et **matplotlib**.

## Résultats et recommandations

L’étude met en évidence que la résiliation des clients est influencée par plusieurs variables contractuelles et comportementales.
Les recommandations issues de cette analyse sont les suivantes :
1. Fidéliser les nouveaux clients par des offres promotionnelles ou des réductions ciblées après les premiers mois d’abonnement.
2. Encourager la souscription à des contrats d’un ou deux ans pour réduire le taux de churn.
3. Surveiller les clients à facture élevée et leur proposer des avantages personnalisés.
4. Améliorer la qualité du service client pour les clients récents, plus susceptibles de résilier.

## Pistes d’amélioration

- Intégrer une méthode d’équilibrage des classes (par exemple **SMOTE**) afin de mieux gérer le déséquilibre entre churners et non-churners.
- Expérimenter des modèles supplémentaires tels que **CatBoost** pour comparaison.
- Développer une **interface Streamlit** afin de permettre l’utilisation interactive du modèle pour la détection du churn en temps réel.
