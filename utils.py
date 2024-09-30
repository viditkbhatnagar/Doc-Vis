import wikipediaapi
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import json

def fetch_wikipedia_page(title):
    headers = {'User-Agent': "dockvis/1.0 (contact: bhatnagar007vidit@gmail.com)"}
    wiki = wikipediaapi.Wikipedia('en', headers=headers)
    page = wiki.page(title)
    
    if page.exists():
        return page.text
    else:
        print("Page not found.")
        return None

def load_model():
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device

def extract_relevant_data(text):
    json_file_name = f"whole_extracted_text.json"
    with open(json_file_name, 'w') as json_file:
        json.dump(text, json_file, indent=4)
    print(f"\033[93mExtracted data saved to {json_file_name}\033[0m")
    

   
    person_pattern = r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b"  
    date_pattern = r"\b\d{1,2} [A-Za-z]+ \d{4}\b"  
    event_pattern = r"\b(\w+ \w+)\b"  

    persons = re.findall(person_pattern, text)
    dates = re.findall(date_pattern, text)
    events = re.findall(event_pattern, text)

   
    extracted_info = []
    for person in persons:
        for date in dates:
            extracted_info.append({
                'person': person,
                'event': 'Relevant Event Here', 
                'date': date
            })

    return extracted_info

def build_knowledge_graph(extracted_data):
    import networkx as nx
    G = nx.Graph()
    for item in extracted_data:
        G.add_node(item['person'], type='person')
        G.add_node(item['event'], type='event')
        G.add_edge(item['person'], item['event'], date=item['date'])
    return G

def main():
    title = input("Enter the Wikipedia title: ")
    document = fetch_wikipedia_page(title)
    
    if document:
        extracted_data = extract_relevant_data(document)
        if extracted_data:
            print("Relevant data extracted:", extracted_data)
            return extracted_data
        else:
            print("No relevant data extracted.")
    else:
        print("Failed to fetch document.")

if __name__ == "__main__":
    main()
