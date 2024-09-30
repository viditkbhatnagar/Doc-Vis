import wikipediaapi
import json
from llm_pipeline import extract_with_llm
import networkx as nx
from pydantic import BaseModel, Field
from typing import List  # Add this import statement

# Ground truth dataset for accuracy calculation
ground_truth = [
    {"person": "John Doe", "event": "Birthday Party", "date": "2022-01-01"},
    {"person": "Jane Smith", "event": "Graduation", "date": "2021-05-20"},
    # Add more entries as needed
]

class ExtractedInfo(BaseModel):
    person: str = Field(description="Name of the person")
    event: str = Field(description="Description of the event")
    date: str = Field(description="Date of the event")

class ExtractedData(BaseModel):
    extracted_info: List[ExtractedInfo] = Field(description="List of extracted information")

    def to_dict(self):
        """Convert the extracted data to a dictionary for JSON serialization."""
        return {
            "extracted_info": [info.dict() for info in self.extracted_info]
        }

def fetch_wikipedia_page(title):
    headers = {'User-Agent': "dockvis/1.0 (contact: bhatnagar007vidit@gmail.com)"}
    wiki = wikipediaapi.Wikipedia('en', headers=headers)
    page = wiki.page(title)
    
    if page.exists():
        return page.text
    else:
        print("Page not found.")
        return None

def extract_relevant_data(text, use_pipeline=True):
    json_file_name = f"whole_extracted_text.json"
    with open(json_file_name, 'w') as json_file:
        json.dump(text, json_file, indent=4)
    print(f"\033[93mExtracted data saved to {json_file_name}\033[0m")
    
    if use_pipeline:
        # Use pipeline-based extraction
        extracted_info = extract_with_llm(text)
    else:
        # Fallback to basic extraction if needed
        extracted_info = [ExtractedInfo(person='Unknown', event='Unknown event', date='Unknown date')]

    return extracted_info

def calculate_accuracy(extracted_data, ground_truth):
    """Calculate accuracy of the extracted data against ground truth."""
    extracted_set = set((item.person, item.event, item.date) for item in extracted_data)
    ground_truth_set = set((item['person'], item['event'], item['date']) for item in ground_truth)
    
    # Calculate true positives
    true_positives = extracted_set.intersection(ground_truth_set)
    
    # Calculate accuracy
    accuracy = len(true_positives) / len(ground_truth_set) * 100 if ground_truth_set else 0
    return accuracy

def build_knowledge_graph(extracted_data):
    """Build a knowledge graph from the extracted data."""
    G = nx.Graph()
    for item in extracted_data:
        G.add_node(item.person, type='person')  # Use dot notation to access attributes
        G.add_node(item.event, type='event')
        G.add_node(item.date, type='date')  # Added date as a node if necessary
        G.add_edge(item.person, item.event, date=item.date)  # Use dot notation
    return G

def main():
    title = input("Enter the Wikipedia title: ")
    document = fetch_wikipedia_page(title)
    
    if document:
        extracted_data = extract_relevant_data(document)
        
        # Ensure extracted_data is not empty before calculating accuracy
        if extracted_data:
            # Calculate and print accuracy
            accuracy = calculate_accuracy(extracted_data, ground_truth)
            print(f"Extraction Accuracy: {accuracy:.2f}%")
            print("Relevant data extracted:", extracted_data)
        else:
            print("No relevant data extracted.")
    else:
        print("Failed to fetch document.")

if __name__ == "__main__":
    main()
