import pandas as pd

def chargement_nettoyage():
      df = pd.read_csv("telco_customer_churn.zip",compression='zip',sep=",")
      
      noms_fr = {
          "customerID": "ID_Client",
          "gender": "Genre",
          "SeniorCitizen": "Client_Senior",
          "Partner": "Partenaire",
          "Dependents": "Personnes_a_charge",
          "tenure": "Anciennete",
          "PhoneService": "Service_Telephone",
          "MultipleLines": "Lignes_multiples",
          "InternetService": "Type_Internet",
          "OnlineSecurity": "Securite_en_ligne",
          "OnlineBackup": "Sauvegarde_en_ligne",
          "DeviceProtection": "Protection_appareil",
          "TechSupport": "Assistance_technique",
          "StreamingTV": "Streaming_TV",
          "StreamingMovies": "Streaming_Films",
          "Contract": "Type_contrat",
          "PaperlessBilling": "Facturation_electronique",
          "PaymentMethod": "Methode_paiement",
          "MonthlyCharges": "Facture_mensuelle",
          "TotalCharges": "Facture_totale",
          "Churn": "Resiliation"
      }

      df = df.rename(columns=noms_fr)
      

      cols_internet = ["Securite_en_ligne", "Sauvegarde_en_ligne", "Protection_appareil",
                 "Assistance_technique", "Streaming_TV", "Streaming_Films"]

      for col in cols_internet:
          df[col] = df[col].replace({"No internet service": "No"})
          df[col] = df[col].map({"Yes": 1, "No": 0})


      df["Lignes_multiples"] = df["Lignes_multiples"].replace({"No phone service": "No"})
      df["Lignes_multiples"] = df["Lignes_multiples"].map({"Yes": 1, "No": 0})

      cols_binaire = ["Partenaire","Personnes_a_charge","Service_Telephone","Facturation_electronique","Resiliation"]

      for col in cols_binaire:
          df[col] = df[col].map({"Yes": 1, "No": 0})

      df["Genre"] = df["Genre"].map({"Female": 1, "Male": 0})

      cols_dum = ["Methode_paiement", "Type_Internet","Type_contrat"]

      df["Type_Internet"] = df["Type_Internet"].replace({"No": "A"})
      
      df = pd.get_dummies(df, columns=cols_dum, drop_first=True,dtype=int)
      
      noms_fr2={
          "Methode_paiement_Bank transfer (automatic)": "Paiement_virement_auto",
          "Methode_paiement_Credit card (automatic)": "Paiement_carte_auto",
          "Methode_paiement_Electronic check": "Paiement_cheque_elec",
          "Methode_paiement_Mailed check": "Paiement_cheque_postal",
          "Type_Internet_Fiber optic": "Fibre_internet",
          "Type_Internet_DSL": "DSL",
          "Type_contrat_One year": "Contrat_1_an",
          "Type_contrat_Two year": "Contrat_2_ans"
      }
      
      df = df.rename(columns=noms_fr2)


      df["Facture_totale"] = pd.to_numeric(df["Facture_totale"], errors="coerce")



      df["Facture_totale"] = df["Facture_totale"].fillna(0)
      
      return df  
