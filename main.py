import data_processing as dp

def main():
    title = input("Enter the Wikipedia title: ")
    text = dp.fetch_wikipedia_page(title)
    if text:
        entities = dp.extract_entities_with_transformer(text)
        G = dp.build_initial_graph(entities)
        dp.visualize_graph(G)
    else:
        print("No content fetched for the given title.")

if __name__ == "__main__":
    main()
