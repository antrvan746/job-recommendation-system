import os
import json

PROJECT_PATH = "d:\My Work\My Subjects\Do an tot nghiep\code\job-recommendation-system"

jdInputPath = os.path.join(PROJECT_PATH, "data\\jds\\jobs.json")
print(jdInputPath)

jdData = []
with open(jdInputPath, encoding="utf-8") as f:
    jdData = json.load(f)
    
    i = 0

    for job in jdData:
        print(i)
        jdData.append(job["skills"] + " " + job['description'])
        i += 1


print(jdData)