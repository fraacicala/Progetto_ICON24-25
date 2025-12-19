import pandas as pd
import numpy as np
import sklearn as sk
from pandas import crosstab
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate;
import os

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

def decisiontree_classifier(X,y):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    model = DecisionTreeClassifier()
    parametri = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }
    grid_search = GridSearchCV(model, parametri, cv=skf)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print(f"Parametri migliori:  {grid_search.best_params_}")
    metriche = ['accuracy_weighted','precision_weighted','recall_weighted','f1_weighted']
    risultati = cross_validate(best_model, X, y, cv=skf, scoring=metriche)
    print(f"Accuracy Media:  {risultati['test_accuracy'].mean():.2%} (+/- {risultati['test_accuracy'].std():.2%})")
    print(f"Precision Media: {risultati['test_precision_weighted'].mean():.2%}")
    print(f"Recall Media:    {risultati['test_recall_weighted'].mean():.2%}")
    print(f"F1-Score Media:  {risultati['test_f1_weighted'].mean():.2%}")
    return best_model












parametri = {
    'n_estimators': [50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}