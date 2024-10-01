import wikipediaapi
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import networkx as nx
import json

# Initialize the Qwen2-7B-Instruct model
model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

def fetch_wikipedia_page(title):
    """ Fetch the content of a Wikipedia page. """
    wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent='docvis/1.0 (your_email@example.com)')
    page = wiki_wiki.page(title)
    return page.text if page.exists() else None

def extract_entities_with_transformer(text):
    """ Extract entities from text using the Qwen2-7B-Instruct model. """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    entities = []
    for idx, label_id in enumerate(predictions[0]):
        label = model.config.id2label[label_id.item()]
        word = tokenizer.decode([inputs.input_ids[0][idx]])
        entities.append({'text': word, 'label': label})
    return entities

def build_initial_graph(entities):
    """ Build a graph from the entities extracted. """
    G = nx.Graph()
    for entity in entities:
        G.add_node(entity['text'], type=entity['label'])
    # Add edges based on some condition here
    return G

def save_data(data, filename):
    """ Save data to a JSON file. """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def visualize_graph(G):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue')
    plt.show()

# Example of using these functions
if __name__ == "__main__":
    # Quick test of the functions
    title = "Apollo 11"
    text = fetch_wikipedia_page(title)
    entities = extract_entities_with_transformer(text)
    G = build_initial_graph(entities)
    visualize_graph(G)
    save_data(entities, 'entities.json')
