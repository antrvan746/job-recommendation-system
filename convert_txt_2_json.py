import json
import os
import sys
import time

from scripts.utils.Cleaner import TextCleaner
from scripts.utils.Extractor import DataExtractor
from scripts.utils.FreqCounter import CountFrequency
from scripts.utils.KeytermExtractor import KeytermExtractor

PROJECT_PATH = "d:\My Work\My Subjects\Do an tot nghiep\code\job-recommendation-system"
INPUT_PATH = "data\\txt_resumes"
SAVE_PATH = "data\\resumes\\resumes2.json"

def extractResumeFromText(data):
    # data = TextCleaner.clean_text(data)
    # print(data)
    # data = TextCleaner.remove_stopwords(data)
    extractor = DataExtractor(data)
    freqCounter = CountFrequency(data)
    keytermExtractor = KeytermExtractor(data)
    res = {
        "particular_words": extractor.extract_particular_words(),
        "entities": extractor.extract_entities(),
        # "pos_frequencies": freqCounter.count_frequency(),
        # "keyterms": keytermExtractor.get_keyterms_based_on_sgrank(),
        "bi_grams": str(keytermExtractor.bi_gramchunker()),
        "tri_grams": str(keytermExtractor.tri_gramchunker())
    }
    return res

def convertTextToJson(inputPath, outputPath):
    dataList = []
    for d in os.listdir(inputPath):
        with open(os.path.join(inputPath, d), "r", encoding="utf-8") as f:
            data = f.read()
            extractedInfo = extractResumeFromText(data)
            dataList.append(extractedInfo)
        print("Extracted", d)
    
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
