# llm_pipeline.py (Updated for GPU support)

from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
import torch

# Load BERT pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

# Set up a pipeline for Named Entity Recognition (NER)
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True, device=device)

def extract_with_llm(text):
    """
    This function uses BERT to extract entities (person, organization, location, etc.)
    from the input text and maps relationships.
    """
    ner_results = nlp_ner(text)

    extracted_entities = []
    for entity in ner_results:
        entity_dict = {
            'entity': entity['entity_group'],
            'word': entity['word'],
            'start': entity['start'],
            'end': entity['end']
        }
        extracted_entities.append(entity_dict)
    
    return extracted_entities

# Example usage:
if __name__ == "__main__":
    sample_text = "John Doe attended the meeting in Paris on 2021-05-20."
    extracted = extract_with_llm(sample_text)
    print(extracted)
