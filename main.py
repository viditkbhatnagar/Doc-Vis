import data_processing as dp
import visualization as viz

def main():
    title = input("Enter the Wikipedia title: ")
    text = dp.fetch_wikipedia_page(title)
    if text:
        entities = dp.extract_entities_with_transformer(text)
        dp.save_data(entities, 'whole_extracted_data.json')  # Save the complete entity extraction
        
        G = dp.build_initial_graph(entities)
        dp.save_data([dict(node, id=n) for n, node in G.nodes(data=True)], 'relevant_extracted_data.json')  # Save graph nodes with attributes
        
        viz.visualize_graph(G)
    else:
        print("Failed to retrieve data. Please check the Wikipedia title and try again.")

if __name__ == "__main__":
    main()
