import json
import matplotlib.pyplot as plt
import networkx as nx
from utils import fetch_wikipedia_page, extract_relevant_data, build_knowledge_graph
from graph_analaysis import analyze_graph, visualize_graph, plot_centrality, find_key_players
from datetime import datetime

def filter_data_by_date(extracted_data, start_year, end_year):
    """Filters the extracted data based on a specific time range."""
    filtered_data = [item for item in extracted_data if start_year <= int(item['date'][-4:]) <= end_year]
    return filtered_data

def limit_entries(data, limit=50):
    """Limit the data to a maximum number of entries."""
    if len(data) > limit:
        print(f"\033[91mWarning: More than {limit} entries found. Only the first {limit} entries will be shown.\033[0m")
        return data[:limit]
    return data

def get_date_range(extracted_data):
    """Finds the earliest and latest dates from the extracted data."""
    dates = [int(item['date'][-4:]) for item in extracted_data]  # Extract years as integers
    return min(dates), max(dates)

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
            
            # Get the date range from extracted data
            min_year, max_year = get_date_range(extracted_data)
            print(f"\033[93mAvailable date range from extracted data: {min_year} to {max_year}\033[0m")
            
            # Ask user for time range within the suggested range
            start_year = int(input(f"Enter the start year (between {min_year} and {max_year}): "))
            end_year = int(input(f"Enter the end year (between {min_year} and {max_year}): "))

            if not (min_year <= start_year <= max_year and min_year <= end_year <= max_year):
                print(f"\033[91mError: Please select a range within the available dates ({min_year} to {max_year}).\033[0m")
                return
            
            # Filter data by the chosen time range
            filtered_data = filter_data_by_date(extracted_data, start_year, end_year)
            
            if not filtered_data:
                print(f"No data available for the period {start_year} to {end_year}.")
                return

            # Limit entries to the first 50
            limited_data = limit_entries(filtered_data, limit=50)

            # Build knowledge graph from limited data
            knowledge_graph = build_knowledge_graph(limited_data)
            
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
