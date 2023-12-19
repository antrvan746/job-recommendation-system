import json
import os
import sys

from scripts.utils.Cleaner import TextCleaner
from scripts.utils.Extractor import DataExtractor

PROJECT_PATH = "d:\My Work\My Subjects\Do an tot nghiep\code\job-recommendation-system"
INPUT_PATH = "data\\txt_resumes"
SAVE_PATH = "data\\resumes\\resumes2.json"

def extractResumeFromText(data):
    data = TextCleaner.clean_text(data)
    extractor = DataExtractor(data)
    res = {
        "name": extractor.extract_names(),
        "email": extractor.extract_emails(),
        "phone": extractor.extract_phone_numbers(),
        "particular_words": extractor.extract_particular_words(),
        "experiences": extractor.extract_experience(),
        "entities": extractor.extract_entities(),
        "resume": data
    }
    return res

def convertTextToJson(inputPath, outputPath):
    dataList = []
    for d in os.listdir(inputPath):
        with open(os.path.join(inputPath, d), "r", encoding="utf-8") as f:
            data = f.read()
            extractedInfo = extractResumeFromText(data)
            dataList.append(extractedInfo)
        print("Extracted ", d)
    
    with open(outputPath, "w") as f:
        json.dump(dataList, f)

    print("----------------------")
    print("Completed")

if __name__ == "__main__":
    inputPath = os.path.join(PROJECT_PATH, INPUT_PATH)
    outputPath = os.path.join(PROJECT_PATH, SAVE_PATH)
    convertTextToJson(inputPath, outputPath)