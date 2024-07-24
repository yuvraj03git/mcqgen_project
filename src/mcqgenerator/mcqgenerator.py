import os
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_community.callbacks import get_openai_callback
import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

key=os.getenv("OPENAI_API_KEY")


llm=ChatOpenAI(openai_api_key=key,model_name="gpt-3.5-turbo-0125",temperature=0.7)

with open("/workspaces/genai_mcq_generator/Response.json","r") as f:
    response_json=json.load(f)

print(response_json)
template="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

quiz_gen_prompt=PromptTemplate(
    input_variables=["text","number","subject","tone","response_json"],
    template=template

)

quiz_chain=LLMChain(llm=llm,prompt=quiz_gen_prompt,output_key="quiz",verbose=True)
new_template="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""
quiz_review_prompt=PromptTemplate(
    input_variables=["subject","quiz"],
    template=new_template

)
quiz_review_chain=LLMChain(llm=llm,prompt=quiz_review_prompt,output_key="review",verbose=True)

genrate_eval_chain=SequentialChain(chains=[quiz_chain,quiz_review_chain],input_variables=["text", "number", "subject", "tone","response_json"],output_variables=["quiz", "review"],verbose=True,)
