from owlready2 import *
import pandas as pd

def load_ontology(path):
    onto = get_ontology(path)
    onto.load()
    return onto

def populate_ontology():
    empty_onto = os.path.join(os.path.dirname(__file__), '..', 'ontology', 'himym_empty.rdf')
    csv_file = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'himym_episodewise.csv')
    df = pd.read_csv(csv_file)
    onto = load_ontology("ddd")
    with onto:
        for i, row in df.iterrows():
            season = str(row["Season"])
            episode = str(row["Episode"])
            ind = "Episode_" + episode + "_" + season
            new_ep = onto.Episode(ind)
        output =  os.path.join(os.path.dirname(__file__), '..', 'ontology', 'himym_populated.rdf')
        onto.save(output)



