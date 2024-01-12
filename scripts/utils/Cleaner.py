from uuid import uuid4
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load the English model
nlp = spacy.load('en_core_web_md')
nltk.download('punkt')
nltk.download('stopwords')

REGEX_PATTERNS = {
    'email_pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'phone_pattern': r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    'link_pattern': r'\b(?:https?://|www\.)\S+\b'
}


def generate_unique_id():
    """
    Generate a unique ID and return it as a string.

    Returns:
        str: A string with a unique ID.
    """
    return str(uuid4())


class TextCleaner:
    """
    A class for cleaning a text by removing specific patterns.
    """

    def remove_emails_links(text):
        """
        Clean the input text by removing specific patterns.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The cleaned text.
        """
        for pattern in REGEX_PATTERNS:
            text = re.sub(REGEX_PATTERNS[pattern], '', text)
        return text

    def clean_text(text):
        """
        Clean the input text by removing specific patterns.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The cleaned text.
        """
        text = TextCleaner.remove_emails_links(text)
        text = TextCleaner.remove_entities(text)
        doc = nlp(text)
        for token in doc:
            if token.pos_ == 'PUNCT':
                text = text.replace(token.text, '')
        return str(text)

    def remove_stopwords(text):
        """
        Clean the input text by removing stopwords.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The cleaned text.
        """
        doc = nlp(text)
        for token in doc:
            if token.is_stop:
                text = text.replace(token.text, '')
        return text

    def remove_custom_words(text, custom_words):
        # Tokenize the text into words
        words = word_tokenize(text)

        # Remove custom words from the tokenized words
        filtered_words = [
            word for word in words if word.lower() not in custom_words]

        # Reconstruct the text without custom words
        filtered_text = " ".join(filtered_words)

        return filtered_text

    def remove_stopwords(text):
        # Tokenize the text into words
        words = word_tokenize(text)

        # Get the list of English stopwords
        stop_words = set(stopwords.words("english"))

        # Remove stopwords from the tokenized words
        filtered_words = [
            word for word in words if word.lower() not in stop_words]

        # Reconstruct the text without stopwords
        filtered_text = " ".join(filtered_words)

        return filtered_text

    def remove_entities(text):
        entities_to_remove = {"GPE", "DATE"}
        # Process the text with SpaCy
        doc = nlp(text)

        # Remove specified entities
        filtered_tokens = [
            token.text for token in doc if token.ent_type_ not in entities_to_remove]

        # Reconstruct the text without the specified entities
        filtered_text = " ".join(filtered_tokens)

        return filtered_text


class CountFrequency:

    def __init__(self, text):
        self.text = text
        self.doc = nlp(text)

    def count_frequency(self):
        """
        Count the frequency of words in the input text.

        Returns:
            dict: A dictionary with the words as keys and the frequency as values.
        """
        pos_freq = {}
        for token in self.doc:
            if token.pos_ in pos_freq:
                pos_freq[token.pos_] += 1
            else:
                pos_freq[token.pos_] = 1
        return pos_freq
