from collections import OrderedDict
import json
import os
import time

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Batch

from sentence_transformers import SentenceTransformer
from numpy.linalg import norm


PROJECT_PATH = "d:\My Work\My Subjects\Do an tot nghiep\code\job-recommendation-system"
RESUME_DATA_PATH = "data\\resumes\\resumes-keybert.json"
JOB_DESC_PATH = "data\\jds\\jobs-keybert.json"
CONFIG_PATH = "scripts\\models\\config.yml"


JOB_SAVE_PATH = "data\\jds\\model-vector.json"
CV_SAVE_PATH = "data\\resumes\\model-vector.json"

def cos_sim(a, b): return (a @ b.T) / (norm(a) * norm(b))


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.max_seq_length = 500


def readDoc(path):
    with open(path) as f:
        try:
            data = json.load(f)
        except Exception as e:
            print("Error when reading JSON file!!!")
            data = {}
    return data

jobsData = readDoc(os.path.join(PROJECT_PATH, JOB_DESC_PATH))
resumeData = readDoc(os.path.join(PROJECT_PATH, RESUME_DATA_PATH))


def calculateMatchingPercentage(resume, jobDescription):
    # Calculate matching percentage
    matching_percentage = cos_sim(resume, jobDescription) * 100
    return matching_percentage


def get_similarity_score2(resume_string, job_description_string):
    resumeEmbed = model.encode(resume_string)
    jdEmbed = model.encode(job_description_string)
    return calculateMatchingPercentage(resumeEmbed, jdEmbed)


def get_vector(text):
    return model.encode(text).tolist()

def saveJson(savePath, data):
    with open(savePath, "w") as f:
        jsonObject = json.dumps(data, sort_keys=True, indent=4)
        f.write(jsonObject)
    


if __name__ == "__main__":
    start = time.time()

    n = 13 # random.randint(0, len(resumeData))
    resume = resumeData[n]["keyterms_keybert"]
    resumeString = ', '.join(map(lambda item: item[0], resume))

    jds = {}

    for i in range(len(jobsData)):
        job = jobsData[i]["keyterms_keybert"]
        jobString = jobsData[i]["title"] + " " + jobsData[i]["skills"] + " "
        jobString += ', '.join(map(lambda item: item[0], job))

        jds[jobsData[i]["title"]] = get_similarity_score2(
            resumeString, jobString)

    ordered_dict = OrderedDict(sorted(jds.items(), key=lambda item: -item[1]))
    end = time.time()
    print(resumeData[n]["position"])
    print("Time train and sort data: ", (end - start) / 60)
    print("-----------------------------------------------------------")

    for key in ordered_dict:
        print(key, "\t\t", ordered_dict[key])


    # n = 2 # random.randint(0, len(jobsData))
    # job = jobsData[n]["keyterms_textrank"]
    # jobString = jobsData[n]["title"] + " " + jobsData[n]["skills"] + ', '.join(map(lambda item: item[0], job))

    # resumes = {}

    # for i in range(len(resumeData)):
    #     resume = resumeData[i]["keyterms_textrank"]
    #     resumeString = ', '.join(map(lambda item: item[0], resume))

    #     resumes[resumeData[i]["position"]] = get_similarity_score2(
    #         resumeString, jobString)

    # ordered_dict = OrderedDict(sorted(resumes.items(), key=lambda item: -item[1]))
    # end = time.time()
    # print(jobsData[n]["title"])
    # print("Time train and sort data: ", (end - start) / 60)
    # print("-----------------------------------------------------------")

    # for key in ordered_dict:
    #     print(key, "\t\t", ordered_dict[key])

    # for i in range(10):
    #     job = jobsData[i]["keyterms_textrank"]
    #     jobString = jobsData[i]["title"] + " " + jobsData[i]["skills"] + ', '.join(map(lambda item: item[0], job))
    #     print(jobsData[i]["title"], "\n", model.encode(jobString))
    #     print()

    # for i in range(10):
    #     resume = resumeData[i]["keyterms_textrank"]
    #     resumeString = ', '.join(map(lambda item: item[0], resume))
    #     print(resumeData[i]["position"], "\n", model.encode(resumeString))
    #     print()

    # data = []
    # for i in range(len(jobsData)):
    #     job = jobsData[i]["keyterms_textrank"]
    #     jobString = jobsData[i]["title"] + " " + jobsData[i]["skills"] + ', '.join(map(lambda item: item[0], job))

    #     tmp = {
    #         "title": jobsData[i]["title"],
    #         "raw": jobString,
    #         "vector": get_vector(jobString)
    #     }

    #     # resume = resumeData[i]["keyterms_textrank"]
    #     # resumeString = ', '.join(map(lambda item: item[0], resume))

    #     # tmp = {
    #     #     "title": resumeData[i]["position"],
    #     #     "raw": resumeString,
    #     #     "vector": get_vector(resumeString)
    #     # }

    #     data.append(tmp)
    # saveJson(os.path.join(PROJECT_PATH, JOB_SAVE_PATH), data)
    
