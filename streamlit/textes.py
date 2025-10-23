intro = """
Dans un secteur des télécommunications particulièrement concurrentiel, la fidélisation des abonnés constitue un enjeu stratégique majeur.  
Acquérir un nouveau client coûte souvent bien plus cher que de conserver un client existant, d’où l’importance d’anticiper les résiliations, ou **churn**, 
afin d’optimiser les actions de rétention et d’améliorer la satisfaction client.

Ce projet vise à **modéliser le risque de résiliation** à partir des données contractuelles et comportementales des abonnés.  
Les informations exploitées incluent l’ancienneté du client, le type de contrat, les services souscrits ou encore le montant des factures.  
Ces variables ont permis d’entraîner un modèle capable d’estimer, pour chaque client, une **probabilité de départ**.

Après plusieurs expérimentations et comparaisons de modèles (détaillées dans le fichier *Projet_churn.ipynb* disponible sur GitHub),  
la **régression logistique** s’est imposée comme la solution la plus pertinente.  
Elle offre un excellent compromis entre **performance**, **stabilité** et **interprétabilité**, 
tout en identifiant clairement les facteurs influençant le churn.

L’application développée ici permet de mettre en pratique ces résultats :  
- Dans la **première page**, vous pouvez consulter les **performances détaillées du modèle** (AUC, précision, rappel, F1-score, courbes ROC et Précision–Rappel).  
- Dans la **deuxième page**, vous pouvez **tester le modèle en temps réel** en entrant les caractéristiques d’un client afin d’obtenir sa probabilité de résiliation.

Cette approche illustre la manière dont la **data science** peut être utilisée pour appuyer les décisions stratégiques 
et orienter les politiques de **fidélisation client** sur la base de données objectives et mesurables.
"""
