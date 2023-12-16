import pandas as pd
import os
import json

PROJECT_PATH = os.getcwd()
INPUT_FILE_PATH = "data\\resumes\\resume.csv"
OUTPUT_FILE_PATH = "data\\resumes\\resumes.json"


def extractResumeInfo(fullName, email, gender, resume):
    resumeInfo = {
        "fullName": fullName,
        "email": email,
        "gender": gender,
        "resume": resume
    }

    return resumeInfo

if __name__ == "__main__":
    filePath = os.path.join(PROJECT_PATH, INPUT_FILE_PATH)
    df = pd.read_csv(filePath)
    print(df)

    dataList = []

    for i in range(len(df)):
        fullName = df['full_name'][i]
        email = df['email'][i]
        gender = df['gender'][i]
        resume = df['resume_str'][i]
        resumeInfo = extractResumeInfo(fullName, email, gender, resume)
        dataList.append(resumeInfo)

    jsonFilePath = os.path.join(PROJECT_PATH, OUTPUT_FILE_PATH)
    with open(jsonFilePath, 'w') as f:
        json.dump(dataList, f)

    