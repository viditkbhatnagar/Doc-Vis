import wikipediaapi
import datetime
import re
import json
from transformers import pipeline
import networkx as nx

def extract_year(date_str):
    """ Extracts the year from a date string. """
    if not date_str:
        return 'No year available'
    try:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return str(date.year)
    except ValueError:
        # Try to extract just the year if the full date isn't present
        match = re.search(r'\b(\d{4})\b', date_str)
        if match:
            return match.group(1)
        return 'No year available'

def fetch_wikipedia_page(title):
    """ Fetch the content of a Wikipedia page using a specified user agent. """
    wiki = wikipediaapi.Wikipedia(language='en', user_agent='docvis/1.0 (bhatnagar007vidit@gmail.com)')
    page = wiki.page(title)
    if page.exists():
        return page.text
    else:
        print("Page not found.")
        return None

def save_data(data, filename):
    """ Save extracted data to a JSON file. """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def extract_entities_with_transformer(text):
    """ Extract entities from text using a transformer-based NER model. """
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    entities = ner_pipeline(text)
    return [{'text': entity['word'], 'type': entity['entity_group'], 'start': entity['start'], 'end': entity['end'], 'date': extract_year(entity['word'])} for entity in entities]

def are_related(entity1, entity2):
    """Determines if two entities should be connected in the graph."""
    # Example condition: entities are of the same type and appear within 100 characters of each other in the text
    if entity1['type'] == entity2['type'] and abs(entity1['start'] - entity2['start']) < 100:
        return True
    return False


def build_initial_graph(entities):
    """Construct a graph from extracted entities, connecting nodes based on shared attributes or defined relationships."""
    G = nx.Graph()
    for entity in entities:
        # You might adjust the handling of 'date' to use 'year' or other modifications as needed
        year = extract_year(entity.get('date', ''))
        G.add_node(entity['text'], type=entity['type'], year=year, start=entity['start'], end=entity['end'])

    # Add edges based on logical relationships
    for i, entity in enumerate(entities):
        for j, other in enumerate(entities):
            if i != j and are_related(entity, other):
                relation_description = f"{entity['text']} and {other['text']} are related due to their type and proximity."
                G.add_edge(entity['text'], other['text'], relation=relation_description)

    return G

