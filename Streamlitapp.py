import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from src.mcqgenerator.mcqgenerator import genrate_eval_chain
from src.mcqgenerator.logger import logging

#Loading json file
with open('Response.json','r') as file:
    RESPONSE_JSON = json.load(file)

#creating a title for the app
st.title("MCQ generator Application with langchain ")

with st.form("user input"):
    uploaded_file=st.file_uploader("upload pdf or text")

    mcq_count=st.number_input("no. of mcq to be generated",min_value=2,max_value=7)
    subject=st.text_input("inser subject",max_chars=20)
    tone=st.text_input("complexity of quiz",max_chars=20,placeholder="simple")
    button=st.form_submit_button("create MCQ's")
    
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text=read_file(uploaded_file)
                #count tokens and the cost of api call
                with get_openai_callback() as cb:
                    response=genrate_eval_chain(
                        {
                            "text":text,
                            "number":mcq_count,
                            "subject":subject,
                            "tone":tone,
                            "response_json":json.dumps(RESPONSE_JSON)
                        }
                    )
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("ERROR")
            else:
                print(f"total tokens:{cb.total_tokens}")
                print(f"Prompt TOkens:{cb.prompt_tokens}")
                print(f"Completion TOkens:{cb.completion_tokens}")
                print(f"Total cost:{cb.total_cost}")
                if isinstance(response,dict):
                    #extract the quiz data from the response
                    quiz=response.get("quiz",None)
                    if quiz is not None:
                        table_data=get_table_data(quiz)
                        if table_data is not None :
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)
                            #display the review in atext box as well
                            st.text_area(label="Review",value=response["review"])
                        else:
                            st.error("Error in table data")

                else:
                        st.write(response)
