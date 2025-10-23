intro = """
Ce projet a pour objectif de **prédire la résiliation des clients (churn)** d’une entreprise de télécommunication 
à partir de leurs données contractuelles, démographiques et comportementales.  
L’enjeu est d’identifier les clients présentant un risque de départ afin de permettre à l’entreprise 
de mettre en place des actions préventives ciblées (fidélisation, offres personnalisées, etc.).

Le modèle choisi est une **régression logistique**, un algorithme statistique simple, robuste et interprétable, 
idéal pour ce type de problématique binaire (résiliation ou non).  
Après un travail de **préparation et de nettoyage des données** (traitement des valeurs manquantes, 
encodage des variables catégorielles, standardisation), plusieurs combinaisons d’hyperparamètres ont été testées 
afin d’optimiser les performances du modèle selon les métriques clés (AUC, précision, rappel, F1-score).

L’application se compose de deux volets :  
- Une **page d’analyse des performances** du modèle, présentant ses résultats et les courbes ROC / Précision–Rappel.  
- Une **interface interactive** permettant de **simuler la probabilité de churn** d’un client selon ses caractéristiques.

Cette approche illustre la manière dont la data science peut être utilisée pour appuyer les décisions stratégiques 
en matière de **rétention client** et d’optimisation des ressources marketing.
"""
