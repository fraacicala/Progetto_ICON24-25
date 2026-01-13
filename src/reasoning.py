from owlready2 import *
import pandas as pd
import os

# Funzione che carica l'ontologia specificata tramite il suo percorso in path.
# Utilizza le funzioni della libreria owlready2.
def load_ontology(path):
    onto = get_ontology(path)
    onto.load()
    return onto

# Funzione che popola l'ontologia con gli individui di tutti gli episodi.
# Estrae i dati dal DataFrame ottenuto tramite pandas.
# Itera su ogni riga e crea un identificatore unico per ogni espisodio, assegnandogli anche il rispettivo titolo.
def populate_ontology():
    empty_onto = os.path.join(os.path.dirname(__file__), '..', 'ontology', 'himym_empty.rdf')
    csv_file = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'himym_episodewise.csv')
    df = pd.read_csv(csv_file)
    onto = load_ontology(empty_onto)
    with onto:
        for i, row in df.iterrows():
            season = str(row["Season"])
            episode = str(row["Episode"]).zfill(2)
            title = str(row["Title"])
            ind = "Episode_" + season + "_" + episode
            new_ep = onto.Episodio(ind)
            new_ep.haTitolo.append(title)
        output =  os.path.join(os.path.dirname(__file__), '..', 'ontology', 'himym_populated.rdf')
        onto.save(output)

# Funzione che esegue il reasoner Hermit, che analizza l'ontologia e inferisce nuova conoscenza basandosi sulle regole scritte in essa.
# Itera su tutti gli episodi e controlla uno ad uno la classe a cui appartengono. Salva la classificazione inferita.
def run_reasoning():
    path_ontology = os.path.join(os.path.dirname(__file__), '..', 'ontology', 'himym_populated.rdf')
    onto = load_ontology(path_ontology)
    with onto:
        sync_reasoner(infer_property_values=True)
    semantic_results = {}
    if hasattr(onto, "Episodio"):
        episode_list = onto.Episodio.instances()
    for ep in episode_list:
        name = ep.name
        if hasattr(onto, "Episodio_eccellente") and ep in onto.Episodio_eccellente.instances():
            target= "Eccellente"

        elif hasattr(onto, "Episodio_scarso") and ep in onto.Episodio_scarso.instances():
            target = "Scarso"

        else:
            target = "Buono"

        semantic_results[name] = target

    print(f" -> Classificazione semantica completata per {len(episode_list)} episodi.")
    return semantic_results




