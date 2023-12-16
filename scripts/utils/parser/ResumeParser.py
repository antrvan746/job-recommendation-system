import json
from utils.Extractor import DataExtractor
from utils.Cleaner import TextCleaner, CountFrequency, generate_unique_id
from utils.KeytermExtractor import KeytermExtractor

SAVE_DIRECTORY = '../../Data/Processed/Resumes'

class ResumeParser:
    def __init__(self, resume: str):
        self.resume_data = resume
        self.clean_data = TextCleaner.clean_text(
            self.resume_data)
        self.entities = DataExtractor(self.clean_data).extract_entities()
        self.experience = DataExtractor(self.clean_data).extract_experience()
        self.years = DataExtractor(self.clean_data).extract_position_year()
        self.key_words = DataExtractor(
            self.clean_data).extract_particular_words()
        self.pos_frequencies = CountFrequency(
            self.clean_data).count_frequency()
        self.keyterms = KeytermExtractor(
            self.clean_data).get_keyterms_based_on_sgrank()
        self.bi_grams = KeytermExtractor(self.clean_data).bi_gramchunker()
        self.tri_grams = KeytermExtractor(self.clean_data).tri_gramchunker()

    def get_JSON(self) -> dict:
        """
        Returns a dictionary of resume data.
        """
        resume_dictionary = {
            "unique_id": generate_unique_id(),
            "resume_data": self.resume_data,
            "clean_data": self.clean_data,
            "entities": self.entities,
            "extracted_keywords": self.key_words,
            "keyterms": self.keyterms,
            "experience": self.experience,
            "years": self.years,
            "bi_grams": str(self.bi_grams),
            "tri_grams": str(self.tri_grams),
            "pos_frequencies": self.pos_frequencies
        }

        return resume_dictionary