import pandas as pd
import os
import learning  # Importa il tuo modulo di learning
import reasoning  # Importa il tuo modulo di reasoning


def main():
    """
    Orchestratore dell'esperimento comparativo ML vs ML+OntoBK.
    """
    print("===================================================================")
    print(" AVVIO PROGETTO: INTEGRAZIONE DI MACHINE LEARNING E CONOSCENZA SEMANTICA")
    print("===================================================================")

    # --- Configurazione dei percorsi ---
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'himym_episodewise.csv')

    # ===================================================================
    #   SCENARIO 1: ESECUZIONE BASELINE (SOLO MACHINE LEARNING)
    # ===================================================================
    print("\n--- [SCENARIO 1] ESECUZIONE MODELLI SU DATASET ORIGINALE ---")

    # 1. Caricamento e preprocessing del dataset originale
    dataset_originale = learning.load_dataset(dataset_path)
    X_orig, y_orig = learning.preprocessing_dataset(dataset_originale.copy())  # Usa una copia per sicurezza

    print("\n[INFO] Addestramento e valutazione del Decision Tree (Baseline)...")
    learning.decisiontree_classifier(X_orig, y_orig)

    print("\n[INFO] Addestramento e valutazione del Random Forest (Baseline)...")
    learning.randomforest_classifier(X_orig, y_orig)

    print("\n--- [SCENARIO 1] COMPLETATO ---")

    # ===================================================================
    #   FASE DI ARRICCHIMENTO CON CONOSCENZA DI FONDO (BK)
    # ===================================================================
    print("\n--- [FASE DI ARRICCHIMENTO] INTEGRAZIONE CONOSCENZA SEMANTICA ---")

    # 1. Popola l'ontologia se non è già stato fatto
    populated_onto_path = os.path.join(os.path.dirname(__file__), 'ontology', 'himym_populated.rdf')
    if not os.path.exists(populated_onto_path):
        print("[INFO] Ontologia popolata non trovata. Avvio il processo di popolamento...")
        reasoning.populate_ontology()
        print("[INFO] Popolamento completato.")
    else:
        print("[INFO] Utilizzo dell'ontologia popolata esistente.")

    # 2. Esegui il ragionamento per ottenere la classificazione semantica
    print("[INFO] Avvio del ragionatore semantico per classificare gli episodi...")
    semantic_results = reasoning.run_reasoning()

    # 3. Converti i risultati semantici in un DataFrame per l'integrazione
    semantic_df = pd.DataFrame(list(semantic_results.items()), columns=['Episode_ID', 'Semantic_Class'])

    # 4. Prepara il dataset originale per il merge
    dataset_arricchito = dataset_originale.copy()
    # Crea una colonna 'Episode_ID' nel DataFrame che corrisponda a quella dell'ontologia
    dataset_arricchito['Episode_ID'] = dataset_arricchito.apply(
        lambda row: f"Episode_{int(row['Season'])}_{int(row['Episode']):02d}", axis=1
    )

    # 5. Unisci il dataset originale con i risultati semantici
    dataset_arricchito = pd.merge(dataset_arricchito, semantic_df, on='Episode_ID', how='left')
    print("\n[SUCCESS] Dataset arricchito con la nuova feature 'Semantic_Class'.")
    # print(dataset_arricchito[['Title', 'Semantic_Class']].head()) # Decommenta per un controllo rapido

    # ===================================================================
    #   SCENARIO 2: ESECUZIONE CON DATASET ARRICCHITO (ML + BK)
    # ===================================================================
    print("\n--- [SCENARIO 2] ESECUZIONE MODELLI SU DATASET ARRICCHITO CON BK ---")

    # 1. Aggiorna il preprocessing per includere la nuova feature
    from sklearn.preprocessing import LabelEncoder

    text_columns = ['Director', 'Writers', 'Verdict', 'Semantic_Class']  # Aggiungi la nuova colonna
    for column in text_columns:
        if column in dataset_arricchito.columns:
            le = LabelEncoder()
            dataset_arricchito[column] = le.fit_transform(dataset_arricchito[column])

    # Definisci le nuove features e il target
    columns_features_arricchite = ['Season', 'Episode', 'Director', 'Writers', 'Viewers', 'Votes',
                                   'Semantic_Class']  # Aggiungi la nuova feature
    X_arr = dataset_arricchito[columns_features_arricchite]
    y_arr = dataset_arricchito['Verdict']

    print("\n[INFO] Addestramento e valutazione del Decision Tree (con BK)...")
    learning.decisiontree_classifier(X_arr, y_arr)

    print("\n[INFO] Addestramento e valutazione del Random Forest (con BK)...")
    learning.randomforest_classifier(X_arr, y_arr)

    print("\n--- [SCENARIO 2] COMPLETATO ---")
    print("\n===================================================================")
    print(" ESPERIMENTO COMPARATIVO CONCLUSO")
    print("===================================================================")


if __name__ == "__main__":
    main()