from typing import Union

from pydantic import BaseModel
from fastapi import FastAPI

from scripts.utils.Cleaner import TextCleaner
from scripts.utils.Extractor import DataExtractor
from scripts.utils.KeytermExtractor import KeytermExtractor

import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

# init params of skill extractor
nlp = spacy.load("en_core_web_lg")
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

app = FastAPI()


class ResumeBody(BaseModel):
    raw_content: str


TOP_N_VALUES = 15
custom_words = []


def extract_skill(obj):
    return {
        "value": obj["doc_node_value"],
        "type": skill_extractor.getSkillType(obj["skill_id"]),
        "score": str(obj["score"])
    }

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/extract-resume")
def extract_resume(resume: ResumeBody):
    data = resume.raw_content
    # data = TextCleaner.remove_stopwords(data)
    # data = TextCleaner.remove_custom_words(data, custom_words)
    # data = TextCleaner.clean_text(data)

    annotations = skill_extractor.annotate(data, tresh=0.8)

    skill_list = []
    fm_res = []
    for item in annotations["results"]["full_matches"]:
        if (item["doc_node_value"] in skill_list):
            continue
        skill_list.append(item["doc_node_value"])
        fm_res.append(extract_skill(item))

    ng_res = []
    for item in annotations["results"]["ngram_scored"]:
        if (item["doc_node_value"] in skill_list):
            continue
        skill_list.append(item["doc_node_value"])
        ng_res.append(extract_skill(item))

    keytermExtractor = KeytermExtractor(data, TOP_N_VALUES)
    res = {
        "keyterms_textrank": keytermExtractor.get_keyterms_based_on_key_bert(),
        "full_matches": fm_res,
        "ngram_scored": ng_res
    }

    return res