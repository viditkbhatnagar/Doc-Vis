import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import re
from typing import List
from pydantic import BaseModel, Field

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

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

def extract_named_entities(text):
    """
    Extract persons from the named entities recognized in the text.
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Apply POS tagging
    tagged_tokens = pos_tag(tokens)
    
    # Perform Named Entity Recognition (NER)
    chunked = ne_chunk(tagged_tokens)
    
    persons = []
    # Traverse the chunked tree and extract named entities tagged as 'PERSON'
    for chunk in chunked:
        if isinstance(chunk, Tree) and chunk.label() == 'PERSON':
            person = ' '.join([token for token, pos in chunk.leaves()])
            persons.append(person)
    
    return persons

def extract_dates(text):
    """
    Extract dates from the text using a simple regular expression pattern.
    """
    date_pattern = r'\b(?:\d{1,2} ?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) ?\d{2,4}|\d{4})\b'
    return re.findall(date_pattern, text)

def extract_events(text):
    """
    Extract basic events from the text. This method looks for sentences that describe events.
    """
    sentences = nltk.sent_tokenize(text)
    events = []
    for sentence in sentences:
        if any(word in sentence.lower() for word in ['occurred', 'happened', 'took place', 'event']):
            events.append(sentence)
    return events[:5]  # Limit to 5 events for brevity

def extract_with_pipeline(text):
    """
    Extract persons, dates, and events using the pipeline.
    """
    persons = extract_named_entities(text)
    dates = extract_dates(text)
    events = extract_events(text)
    
    extracted_info = []
    # Ensure we don't go out of range by using the minimum length of the lists
    for i in range(min(len(persons), len(dates), len(events))):
        extracted_info.append(ExtractedInfo(
            person=persons[i] if i < len(persons) else "Unknown",
            event=events[i] if i < len(events) else "Unknown event",
            date=dates[i] if i < len(dates) else "Unknown date"
        ))
    
    return ExtractedData(extracted_info=extracted_info)

def extract_with_llm(text):
    """
    Extract information using the LLM-compatible pipeline.
    This function is kept for compatibility with the rest of the codebase.
    """
    return extract_with_pipeline(text).extracted_info
