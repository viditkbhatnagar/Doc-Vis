import json
import matplotlib.pyplot as plt
import networkx as nx
from utils import fetch_wikipedia_page, extract_relevant_data, build_knowledge_graph
from graph_analaysis import analyze_graph, visualize_graph, plot_centrality, find_key_players

def main():
    title = input("\033[93mEnter the Wikipedia title: \033[0m")
    document = fetch_wikipedia_page(title)
    
    if document:
        extracted_data = extract_relevant_data(document)
        
        if extracted_data:
            print("\033[92mRelevant data extracted\033[0m")
            json_file_name = f"{title.replace(' ', '_')}_extracted_data.json"
            with open(json_file_name, 'w') as json_file:
                json.dump(extracted_data, json_file, indent=4)
            print(f"\033[92mExtracted data saved to {json_file_name}\033[0m")
            
            # Build knowledge graph
            knowledge_graph = build_knowledge_graph(extracted_data)
            
            # Visualize graph
            visualize_graph(knowledge_graph)
            
            # Compute degree centrality (or any other centrality measure)
            centrality_data = nx.degree_centrality(knowledge_graph)
            
            # Plot the centrality data
            plot_centrality(centrality_data)
        else:
            print("No relevant data extracted.")
    else:
        print("No document found.")

if __name__ == "__main__":
    main()
