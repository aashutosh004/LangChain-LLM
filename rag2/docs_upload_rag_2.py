import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

st.title("Conversational RAG with PDF, DOCX, and TXT Support")
st.write("Upload PDF, DOCX, or TXT files and chat with their contents")

session_id = st.text_input("Session ID", value="default_session")
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_path)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue

        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question "
                   "which might reference context in the chat history, "
                   "formulate a standalone question which can be understood "
                   "without the chat history. Do NOT answer the question, "
                   "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. "
                   "Use the following pieces of retrieved context to answer "
                   "the question. If you don't know the answer, say that you "
                   "don't know. Use three sentences maximum and keep the "
                   "answer concise.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)
