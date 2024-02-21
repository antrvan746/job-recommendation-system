import time
import json
import os
from scripts.utils.Cleaner import TextCleaner
from scripts.utils.Extractor import DataExtractor
from scripts.utils.KeytermExtractor import KeytermExtractor

PROJECT_PATH = "d:\My Work\My Subjects\Do an tot nghiep\code\job-recommendation-system"
INPUT_PATH = "data\\test_raw\\test.txt"
SAVE_PATH = "data\\resumes\\resumes-test-multi-rank.json"

TOP_N_VALUES = 20

custom_words = ["reasons", "to", "join", "salary", "loyalty", "bonus", "additional", "health", "insurance", "attractive", "net",  "package", "yearly", "premium", "kpi", "opportunity", "job", "work", "allowance", "paid", "leave", "responsible", "experience", "year", "day", "budget", "month", "months", "time", "work", "project", "month", "nbsp;", "years", "day", "%", "nbsp", "&", "\u200b"]

def extractResumeFromText(data):
    # data = TextCleaner.clean_text(data)
    # data = TextCleaner.remove_stopwords(data)
    # data = TextCleaner.remove_custom_words(data, custom_words)
    keytermExtractor = KeytermExtractor(data, 20)
    
    res = {
        # "particular_words": extractor.extract_particular_words(),
        # "entities": extractor.extract_entities(),
        # "pos_frequencies": freqCounter.count_frequency(),
        # "keyterms": keytermExtractor.get_keyterms_based_on_sgrank(),
        # "keyterms_textrank" : keytermExtractor.get_keyterms_based_on_textrank(),
        # "keyterms_scake": keytermExtractor.get_keyterms_based_on_scake(),
        # "keyterms_yake" : keytermExtractor.get_keyterms_based_on_yake(),
        # "bi_grams": str(keytermExtractor.bi_gramchunker()),
        # "tri_grams": str(keytermExtractor.tri_gramchunker()),
        # "keyterms_rakun2" : keytermExtractor.get_keyterms_based_on_rakun2(),
        "keyterms_keybert" : keytermExtractor.get_keyterms_based_on_key_bert(),
        # "keyterms_multi_rank" : keytermExtractor.get_keyterms_based_on_multi_rank(),
    }
    return res

def convertTextToJson(inputPath, outputPath):
    dataList = []
    with open(inputPath, "r", encoding="utf-8") as f:
        data = f.read()
        data = ' '.join(data.split())
        extractedInfo = extractResumeFromText(data)
        dataList.append(extractedInfo)
    print("Extracted")
    
    with open(outputPath, "w") as f:
        jsonObject = json.dumps(dataList, sort_keys=True, indent=14)
        f.write(jsonObject)


if __name__ == "__main__":
    inputPath = os.path.join(PROJECT_PATH, INPUT_PATH)
    outputPath = os.path.join(PROJECT_PATH, SAVE_PATH)
    start = time.time()
    convertTextToJson(inputPath, outputPath)
    end = time.time()
    print((end - start)/ 60)