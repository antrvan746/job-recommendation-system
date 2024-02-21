# imports
import spacy
from spacy.matcher import PhraseMatcher
import json

# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

# init params of skill extractor
nlp = spacy.load("en_core_web_lg")
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# extract skills from job_description
job_description = """
Akhil                                                                              
Sr. Business Systems Analyst
akhil.mohan0109@gmail.com
Phone no: 510-953-0677
Professional Summary:
8+ years of intensifying experience in multiple roles as Business Analyst, Business Systems Analyst, Scrum Master and achieved titles like Modern Analyst, Organizational Analyst with excellent understanding of various software development life cycle(SDLC)  methodologies such as Waterfall, Agile, Hybrid Waterfall-Scrum framework and processes with good domain knowledge in Banking, Finance and E-commerce
Fine knowledge and comprehension of different software development methodologies such as Kanban and Scrumban, XP(extreme programming), Rational Unified Process (RUP), Scaled Agile Framework (SAFe)
Highly-motivated, Innovative, Skilled-Listener, Excellent Negotiation, Proactive, Quick-Learning Individual
Certified Scrum Master with immense skills in facilitating the Scrum Ceremonies, User Story Workshops, Training the teams to better understand Scrum and increase the teams overall productivity
Strong Leadership skills in handling multiple teams and Offshore teams and ability to effectively communicate with senior management, third party vendors, technical staff and Business users to improve business value
Excellent analytical skills to understand the Business process, functionality, cross functional requirements across various business units and translating them into requirement specifications in order to provide comprehensive solutions and understanding of project process and ability to analyze business problems and identify solutions
Experienced in conducting As-Is and To-Be (Gap Analysis) analysis and possess strong knowledge in carrying out processes for Risk Analysis, SWOT (strength weakness opportunity and threat) Analysis, Cluster Analysis, Change Management and perform Impact Analysis to assess the change and feasibility study
Ability to work under tight deadlines and multi-tasking to meet business objectives, scheduling meetings, negotiating and coordinating with software developers, solution architects and QA teams
Proficient in writing user stories (INVEST format) and handling the requirement churn. Efficient at facilitating Estimation techniques such as Planning Poker, T-shirt sizing, Relative Mass Valuation and Prioritization techniques such as MoSCoW method, Kano techniques and Business Value Based
Engaged with Product Owners to successfully break down Epics into User Stories with INVEST Technique and helped the Scrum Team finalize Tasks for Sprint Backlog using SMART Technique
Extensive Expertise in creating various artifacts including Request for Proposal (RFP),Business Requirement Document(BRD), Product Requirement document (PRD),Software Requirement Specification(SRS), Functional Requirement Document(FRD),Test Plan, Test Scenarios and Test cases as well as documenting project processes and procedures
Extensive experience with Bloomberg Trading System, Aladdin and Charles River Trading.
Hands on various tools such as MS Word, MS Excel, MS PowerPoint, Atlassian  Jira, Team Foundation Server (TFS), HP Agile Manager and MS Project Professional for planning, tracking and managing projects
Managed requirements and tracked defects working with HP Application Lifecycle Management(HPALM) and HP Quality Center (HPQC) and well versed in conducting various types of testing including Smoke, Sanity Testing, Regression Testing, System Testing and User Acceptance Testing (UAT)and documented performance reports
"""

annotations = skill_extractor.annotate(job_description, tresh=0.8)

# with open("savefile.json", "w") as f:
#     jsonObject = json.dumps(annotations, sort_keys=True, indent=14)
#     f.write(jsonObject)

def print_key(obj):
    return "value: " + obj["doc_node_value"] + " type: " + skill_extractor.getSkillType(obj["skill_id"]) + " score: " + str(obj["score"])

print("----------------Full--------------------------")
res = list(map(lambda x: print_key(x), annotations["results"]["full_matches"]))
for i in res:
    print(i)

print("------------------Skills----------------------")
res = list(map(lambda x: print_key(x), annotations["results"]["ngram_scored"]))
for i in res:
    print(i)