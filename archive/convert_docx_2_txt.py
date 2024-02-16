from docx import Document
import io
import shutil
import os

PROJECT_PATH = "d:\My Work\My Subjects\Do an tot nghiep\code\job-recommendation-system"
INPUT_PATH = "data\\raw_resumes2"
SAVE_PATH = "data\\txt_resumes"

def convertDocxToText(inputPath, outputPath):
    for d in os.listdir(inputPath):
        fileExt = d.split(".")[-1]

        if (fileExt == "docx"):
            docxFilename = os.path.join(inputPath, d)
            document = Document(docxFilename)
            textFilename = os.path.join(PROJECT_PATH, SAVE_PATH, d.split(".")[0] + ".txt")
            with open(textFilename, "w", encoding="utf-8") as textFile:
                for para in document.paragraphs: 
                    textFile.write(para.text + "\n")
            print("Completed " + docxFilename)
    print("-------------------------------------")
    print("Completed")

if __name__ == "__main__":
    inputPath = os.path.join(PROJECT_PATH, INPUT_PATH)
    convertDocxToText(inputPath, SAVE_PATH)