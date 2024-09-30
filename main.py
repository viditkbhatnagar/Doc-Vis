import matplotlib.pyplot as plt
from utils import fetch_wikipedia_page, extract_relevant_data, build_knowledge_graph
from graph_analaysis import analyze_graph, visualize_graph, plot_centrality, find_key_players
#vidit
def main():
    title = input("Enter the Wikipedia title: ")
    document = fetch_wikipedia_page(title)
    
    if document:
        extracted_data = extract_relevant_data(document)
        
        if extracted_data:
            print("Relevant data extracted:", extracted_data)

            # Build the knowledge graph
            G = build_knowledge_graph(extracted_data)
            visualize_graph(G)  # Visualize the knowledge graph

            # Analyze the graph
            analysis_results = analyze_graph(G)

            # Plot centrality measures
            plot_centrality(analysis_results['degree_centrality'])

            key_players = find_key_players(G)
            print("Key players in the network:", key_players)
        else:
            print("No relevant data extracted.")
    else:
        print("Failed to fetch document.")

if __name__ == "__main__":
    main()
