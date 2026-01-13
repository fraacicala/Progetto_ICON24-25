import pandas as pd
import os
import learning
import reasoning
from sklearn.preprocessing import LabelEncoder

# Funzione main.
# Nella prima parte richiama le funzioni del modulo learning che permettono il caricamento e pre-processing del dataset e
# l'addestramento del Decision Tree e Random Forest, con la stampa delle relative metriche e grafici.
# Nella seconda parte richiama le funzioni del modulo reasoning per il caricamento e popolamento dell'ontologia e
# l'esecuzione del reasoner Hermit. Crea nuova feature "Semantic_class".
# Nella terza parte riesegue l'addestramento dei modelli, ma sul dataset arricchito da conoscenza semantica.
def main():
    print("AVVIO PROGETTO: ")
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'himym_episodewise.csv')

    print("\n1° parte: esecuzione modelli sul dataset originale...")

    dataset_originale = learning.load_dataset(dataset_path)
    X_orig, y_orig = learning.preprocessing_dataset(dataset_originale.copy())

    print("\nAddestramento e valutazione del Decision Tree...")
    dt_model_orig = learning.decisiontree_classifier(X_orig, y_orig)

    print("\nAddestramento e valutazione del Random Forest...")
    rf_model_orig = learning.randomforest_classifier(X_orig, y_orig)

    print("\n 1° parte completata.")

    print("\n 2° parte: esecuzione ragionamento automatico...")

    populated_onto_path = os.path.join(os.path.dirname(__file__), 'ontology', 'himym_populated.rdf')
    if not os.path.exists(populated_onto_path):
        print("[Ontologia popolata non trovata. Avvio il processo di popolamento...")
        reasoning.populate_ontology()
        print("Popolamento completato.")
    else:
        print("Utilizzo dell'ontologia popolata esistente.")
    print("Avvio del ragionatore semantico per classificare gli episodi...")
    semantic_results = reasoning.run_reasoning()
    semantic_df = pd.DataFrame(list(semantic_results.items()), columns=['Episode_ID', 'Semantic_Class'])
    dataset_arricchito = dataset_originale.copy()
    dataset_arricchito['Episode_ID'] = dataset_arricchito.apply(
        lambda row: f"Episode_{int(row['Season'])}_{int(row['Episode']):02d}", axis=1
    )
    dataset_arricchito = pd.merge(dataset_arricchito, semantic_df, on='Episode_ID', how='left')
    print("\nDataset arricchito con la nuova feature 'Semantic_Class'.")

    print("\n 2° parte completata.")

    print("\n3° parte: esecuzione dei modelli su dataset arricchito con conoscenza di fondo...")
    dataset_arricchito['Semantic_Class'] = dataset_arricchito['Semantic_Class'].fillna(
        'Non Anotato')
    text_columns = ['Director', 'Writers', 'Verdict', 'Semantic_Class']
    for column in text_columns:
        if column in dataset_arricchito.columns:
            le = LabelEncoder()
            dataset_arricchito[column] = le.fit_transform(dataset_arricchito[column])
    columns_features_arricchite = ['Season', 'Episode', 'Director', 'Writers', 'Viewers', 'Votes',
                                   'Semantic_Class']
    X_arr = dataset_arricchito[columns_features_arricchite]
    y_arr = dataset_arricchito['Verdict']

    print("\nAddestramento e valutazione del Decision Tree (con BK)...")
    dt_model_arr = learning.decisiontree_classifier(X_arr, y_arr)

    print("\n[INFO] Addestramento e valutazione del Random Forest (con BK)...")
    rf_model_arr = learning.randomforest_classifier(X_arr, y_arr)

    print("\n 3° parte completata. Esperimento concluso.")

if __name__ == "__main__":
    main()