from collections import OrderedDict
import json
import os
import time

import cohere
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Batch

from sentence_transformers import SentenceTransformer
from numpy.linalg import norm


PROJECT_PATH = "d:\My Work\My Subjects\Do an tot nghiep\code\job-recommendation-system"
RESUME_DATA_PATH = "data\\resumes\\resumes2.json"
JOB_DESC_PATH = "data\\jds\\jobs2.json"
CONFIG_PATH = "scripts\\models\\config.yml"

cos_sim = lambda a,b : (a @ b.T) / (norm(a) * norm(b))

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

# def readConfig(path):
#     try:
#         with open(path) as f:
#             config = yaml.safe_load(f)
#         return config
#     except Exception as e:
#         print("Error when reading file")
#     return None

jobsData = readDoc(os.path.join(PROJECT_PATH, JOB_DESC_PATH))
resumeData = readDoc(os.path.join(PROJECT_PATH, RESUME_DATA_PATH))

class QdrantSearch:
    def __init__(self, resumes, jd):
        # config = readConfig(os.path.join(PROJECT_PATH, CONFIG_PATH))
        self.cohere_key = "fx9JHiH5AqOnM1v9X3o0na1Ra1yIAbK1lk5VHhqA"
        self.qdrant_key = "dlOX5NbYC3DMM3LhtihM1DeUabvTDaIiryAewmtQxCIrHynZFTJ0VQ"
        self.qdrant_url = "https://dcbe9d88-0477-4e78-b5cc-95a2124f61bd.us-east4-0.gcp.cloud.qdrant.io:6333"
        self.resumes = resumes
        self.jd = jd
        self.cohere = cohere.Client(self.cohere_key)
        self.collection_name = "resume_collection_name"
        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_key,
        )

        vector_size = 4096
        print(f"collection name={self.collection_name}")
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

    def get_embedding(self, text):
        try:
            embeddings = self.cohere.embed([text], "large").embeddings
            return list(map(float, embeddings[0])), len(embeddings[0])
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {e}", exc_info=True)

    def update_qdrant(self):
        vectors = []
        ids = []
        for i, resume in enumerate(self.resumes):
            vector, size = self.get_embedding(resume)
            vectors.append(vector)
            ids.append(i)
        try:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=[{"text": resume} for resume in self.resumes]

                )
            )
        except Exception as e:
            self.logger.error(f"Error upserting the vectors to the qdrant collection: {e}", exc_info=True)

    def search(self):
        vector, _ = self.get_embedding(self.jd)

        hits = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=30
        )
        results = []
        for hit in hits:
            result = {
                'text': str(hit.payload)[:30],
                'score': hit.score
            }
            results.append(result)

        return results


def get_similarity_score(resume_string, job_description_string):
    # print("Started getting similarity score")
    qdrant_search = QdrantSearch([resume_string], job_description_string)
    qdrant_search.update_qdrant()
    search_result = qdrant_search.search()
    # print("Finished getting similarity score")
    return search_result

def calculateMatchingPercentage(resume, jobDescription):
    # Calculate matching percentage
    matching_percentage = cos_sim(resume, jobDescription) * 100
    return matching_percentage

def get_similarity_score2(resume_string, job_description_string):
    resumeEmbed = model.encode(resume_string)
    jdEmbed = model.encode(job_description_string)
    return calculateMatchingPercentage(resumeEmbed, jdEmbed)


if __name__ == "__main__":
    start = time.time()
    # resume = resumeData[0]["particular_words"]
    # resumeString = ' '.join(resume)

    resume = resumeData[0]["bi_grams"] + resumeData[0]["tri_grams"]
    resumeString = resume

    jds = {}

    for i in range(len(jobsData)): # len(jobsData)):
        # job = jobsData[i]["particular_words"]
        # jobString = ' '.join(job)
        job = jobsData[i]["bi_grams"] + jobsData[i]["tri_grams"]
        jobString = job
        jds[jobsData[i]["title"]] = get_similarity_score2(resumeString, jobString)

    ordered_dict = OrderedDict(sorted(jds.items(), key=lambda item: -item[1]))
    end = time.time()
    print("Time train and sort data: ", (end - start)/ 60)
    print("-----------------------------------------------------------")

    for key in ordered_dict:
        print(key, "\t\t", ordered_dict[key])