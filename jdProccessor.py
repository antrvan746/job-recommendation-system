import json
import os
import sys
import time
from langdetect import detect

from scripts.utils.Cleaner import TextCleaner
from scripts.utils.Extractor import DataExtractor
from scripts.utils.FreqCounter import CountFrequency
from scripts.utils.KeytermExtractor import KeytermExtractor

PROJECT_PATH = "d:\My Work\My Subjects\Do an tot nghiep\code\job-recommendation-system"
INPUT_PATH = "data\\jds\\jobs.json"
SAVE_PATH = "data\\jds\\jobs2.json"

custom_words = ["reasons", "to", "join", "salary", "loyalty", "bonus", "additional", "health", "insurance", "attractive", "net",  "package", "yearly", "premium", "kpi", "opportunity", "job", "work", "allowance", "paid", "leave", "responsible", "experience", "year", "day", "budget", "month", "months", "time", "work", "project", "month"]

def extractJobDescriptionFromJobJSON(data):
    data["description"] = TextCleaner.remove_entities(data["description"])
    data["description"] = TextCleaner.remove_custom_words(data["description"], custom_words)
    data["description"] = TextCleaner.remove_stopwords(data["description"])
    # data = TextCleaner.remove_stopwords(data)
    extractor = DataExtractor(data["description"])
    # freqCounter = CountFrequency(data["description"])
    # particular_words = ' '.join(extractor.extract_particular_words());
    keytermExtractor = KeytermExtractor(data["description"])
    # keytermExtractor2 = KeytermExtractor(particular_words)
    res = {
        "title": data["title"],
        "industry": data["industry"],
        "skills": data["skills"],
        "companyName": data["companyName"],
        "employmentType": data["employmentType"],
        "entities": extractor.extract_entities(),
        # "pos_frequencies": freqCounter.count_frequency(),
        "particular_words": extractor.extract_particular_words(),
        "keyterms": keytermExtractor.get_keyterms_based_on_sgrank(),
        "bi_grams": str(keytermExtractor.bi_gramchunker()),
        "tri_grams": str(keytermExtractor.tri_gramchunker())
    }
    return res

def convertTextToJson(inputPath, outputPath):
    dataList = []
    with open(inputPath, "r", encoding="utf-8") as f:
        data = json.load(f)
        for job in data:
            if (detect(job["description"]) != "en"):
                continue
            extractedInfo = extractJobDescriptionFromJobJSON(job)
            dataList.append(extractedInfo)
            print("Extracted", job["title"])
    
    with open(outputPath, "w") as f:
        jsonObject = json.dumps(dataList, sort_keys=True, indent=14)
        f.write(jsonObject)

    print("----------------------")
    print("Completed")

if __name__ == "__main__":
    inputPath = os.path.join(PROJECT_PATH, INPUT_PATH)
    outputPath = os.path.join(PROJECT_PATH, SAVE_PATH)
    start = time.time()
    convertTextToJson(inputPath, outputPath)
    end = time.time()
    print((end - start)/ 60)
