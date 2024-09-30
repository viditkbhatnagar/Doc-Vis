import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from collections import Counter
from data_processing import extract_topics

def plot_timeline(data):
    dates = [item['date'] for item in data]
    events = [item['event'] for item in data]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, [1] * len(dates), 'ro-')
    for i, event in enumerate(events):
        plt.text(dates[i], 1.02, event, rotation=45, ha='right', fontsize=8)
    plt.title("Event Timeline")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

def plot_person_relations(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    person_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'person']
    event_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'event']

    nx.draw_networkx_nodes(G, pos, nodelist=person_nodes, node_color='lightblue', node_size=3000, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=event_nodes, node_color='lightgreen', node_size=2000, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

    plt.title("Person-Event Relationship Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_entity_distribution(data):
    entity_types = ['person', 'event']
    entity_counts = [len(set(item['person'] for item in data)), len(set(item['event'] for item in data))]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=entity_types, y=entity_counts)
    plt.title("Distribution of Entities")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.show()

def plot_topic_distribution(text):
    topics = extract_topics(text)
    topic_weights = [weight for _, weight in topics]
    topic_labels = [f"Topic {i+1}" for i in range(len(topics))]

    plt.figure(figsize=(10, 6))
    plt.pie(topic_weights, labels=topic_labels, autopct='%1.1f%%', startangle=90)
    plt.title("Topic Distribution")
    plt.axis('equal')
    plt.show()