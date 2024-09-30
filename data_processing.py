import json
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def process_document(text):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    processed_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(processed_tokens)

def extract_entities(result):
    try:
        # Try to parse the entire result as JSON
        extracted_data = json.loads(result)
        if isinstance(extracted_data, list):
            return extracted_data
        else:
            return [extracted_data]
    except json.JSONDecodeError:
        # If parsing fails, try to extract individual JSON objects
        extracted_data = []
        for line in result.split('\n'):
            try:
                obj = json.loads(line.strip())
                extracted_data.append(obj)
            except json.JSONDecodeError:
                continue
        
        if not extracted_data:
            print("Error parsing the extracted data.")
        
        return extracted_data

def build_knowledge_graph(extracted_data):
    G = nx.Graph()
    for item in extracted_data:
        G.add_node(item['person'], type='person')
        G.add_node(item['event'], type='event')
        G.add_edge(item['person'], item['event'], date=item['date'])
    return G

def extract_topics(text, num_topics=5):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    processed_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    dictionary = corpora.Dictionary([processed_tokens])
    corpus = [dictionary.doc2bow(processed_tokens)]

    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    
    topics = lda_model.print_topics(num_words=5)
    return topics
