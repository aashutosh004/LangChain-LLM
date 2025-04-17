from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain_ollama import OllamaLLM
import uvicorn
import os
from dotenv import load_dotenv

## initialize environment variable
load_dotenv()

## Langsmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY']  = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "A simple API Server"
)

model1 = OllamaLLM(model = 'llama3.2:1b')
model2 = OllamaLLM(model='moondream')

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words")

add_routes(
    app,
    prompt1|model1,
    path = "/essay"
)

add_routes(
    app,
    prompt2|model2,
    path = '/poem'
)

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)