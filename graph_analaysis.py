import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def analyze_graph(G):
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

    # Find communities
    communities = list(nx.community.greedy_modularity_communities(G))

    return {
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'eigenvector_centrality': eigenvector_centrality,
        'communities': communities
    }

def cluster_events(events, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(events)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    return kmeans.labels_

def find_key_players(G):
    degree_centrality = nx.degree_centrality(G)
    key_players = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:5]
    return key_players


def visualize_graph(G):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    plt.title("Knowledge Graph")
    plt.axis('off')
    plt.show()

def plot_centrality(centrality_data):
    plt.figure(figsize=(12, 6))
    labels = list(centrality_data.keys())
    values = list(centrality_data.values())
    plt.bar(labels, values, color='lightcoral')
    plt.title("Centrality Measures")
    plt.ylabel("Centrality Score")
    plt.xticks(rotation=45)
    plt.show()
