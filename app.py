import streamlit as st
import logging
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate

custom_prompt_template = """
You are an AI Coding Assistant, and your task is to solve coding problems and return code snippets based on the given user's query. Below is the user's query.
Query: {query}

Helpful Answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['query'])
    return prompt

# Loading the model with the correct repo ID and repo type
def load_model():
    repo_type = "Zeek98"  # Use "username" for user-owned models
    repo_id = "Zeek98/CodeLLama"  # Replace with the correct repo ID
    
    llm = CTransformers(
        model=repo_id,
        repo_type=repo_type,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.05,
        repetition_penalty=1.4
    )
    return llm

def chain_pipeline():
    llm = load_model()
    qa_prompt = set_custom_prompt()
    qa_chain = LLMChain(prompt=qa_prompt, llm=llm)
    return qa_chain

llmchain = chain_pipeline()

def bot(query):
    llm_response = llmchain.run({"query": query})
    return llm_response

# Create a Streamlit app
st.set_page_config(page_title='Code Llama Demo', page_icon="🐐", layout='wide', initial_sidebar_state='auto')

st.title('Code Llama Bot')

query = st.text_area("Enter your coding query:")
if st.button("Submit"):
    if query:
        try:
            # Log the user's query
            logging.info(f"User query: {query}")

            bot_message = bot(query)

            # Log the AI's response
            logging.info(f"AI response: {bot_message}")

            st.write("AI Response:")
            st.write(bot_message)
        except Exception as e:
            # Log errors
            logging.error(f"Error: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query.")

# Add a footer or additional information as needed
