import pandas as pd
import numpy as np
import sklearn as sk
from pandas import crosstab
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
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
    model = DecisionTreeClassifier(random_state=42)
    parametri = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 5]
    }
    grid_search = GridSearchCV(model, parametri, cv=skf)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print(f"Parametri migliori:  {grid_search.best_params_}")
    metriche = ['accuracy','precision_weighted','recall_weighted','f1_weighted']
    risultati = cross_validate(best_model, X, y, cv=skf, scoring=metriche)
    print(f"Accuracy:  {risultati['test_accuracy'].mean():.2%} (+/- {risultati['test_accuracy'].std():.2%})")
    print(f"Precision Media: {risultati['test_precision_weighted'].mean():.2%}")
    print(f"Recall Media:    {risultati['test_recall_weighted'].mean():.2%}")
    print(f"F1-Score Media:  {risultati['test_f1_weighted'].mean():.2%}")
    mostra_curva_apprendimento(best_model, X, y, title="Learning Curve - Decision Tree")
    return best_model

def randomforest_classifier(X,y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(random_state= 42, class_weight='balanced')
    parametri = {
        'n_estimators': [50, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, ]
    }
    grid_search = GridSearchCV(model, parametri, cv=skf)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print(f"Parametri migliori:  {grid_search.best_params_}")
    metriche = ['accuracy','precision_weighted','recall_weighted','f1_weighted']
    risultati = cross_validate(best_model, X, y, cv=skf, scoring=metriche)
    print(f"Accuracy:  {risultati['test_accuracy'].mean():.2%} (+/- {risultati['test_accuracy'].std():.2%})")
    print(f"Precision Media: {risultati['test_precision_weighted'].mean():.2%}")
    print(f"Recall Media:    {risultati['test_recall_weighted'].mean():.2%}")
    print(f"F1-Score Media:  {risultati['test_f1_weighted'].mean():.2%}")
    mostra_curva_apprendimento(best_model, X, y, title="Learning Curve - Random Forest")
    return best_model


def mostra_curva_apprendimento(estimator, X, y, title="Curva di Apprendimento"):
    print(f"Generazione grafico per: {title}...")

    # Calcola i punti della curva
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
    )

    # Calcola media e deviazione standard
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Disegna il grafico
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Numero di esempi di training")
    plt.ylabel("Accuratezza (Score)")
    plt.grid()

    # Disegna l'area colorata (la varianza)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # Disegna le linee
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score (Test)")

    plt.legend(loc="best")
    plt.show()  # Mostra il grafico a schermo


def main():
    filename = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'himym_episodewise.csv')
    dataset = load_dataset(filename)
    X, y = preprocessing_dataset(dataset)
    decisiontree_model=decisiontree_classifier(X,y)
    randomforest_model=randomforest_classifier(X,y)

if __name__ == "__main__":
    main()