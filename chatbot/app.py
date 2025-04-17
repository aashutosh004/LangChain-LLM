from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

import streamlit as st
import os
from dotenv import load_dotenv

## LangSmith Tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## Streamlit framework
st.title('LangChain Demo With OpenSource LLama')
input_text = st.text_input("Search the Topic you want")

## LLM
llm = OllamaLLM(model="llama3.2:1b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))