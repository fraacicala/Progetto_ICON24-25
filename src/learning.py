import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder

#Funzione per caricare il dataset specificato utilizzando il suo percorso.
#Utilizza la libreria pandas per leggere il file csv.
def load_dataset(path):
    df = pd.read_csv(path)
    return df

#Funzione per effettuare un preprocessing sui dati.
#Trasforma le colonne testuali in valori numerici utilizzando LabelEncoder.
#Definisce le features (X) e il target (y).
def preprocessing_dataset(df):
    text_columns = ['Director', 'Writers', 'Verdict']
    for column in text_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    columns_features = ['Season','Episode','Director', 'Writers','Viewers','Votes']
    X = df[columns_features]
    y=df['Verdict']
    return X, y