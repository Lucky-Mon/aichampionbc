# rag_qa_streamlit_app/rag_pipeline.py

import os
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
vectorstore = None
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss").lower()

def initialize_vectorstore():
    global vectorstore
    dummy_doc = ["This is a placeholder document."]
    if VECTOR_STORE_TYPE == "chroma":
        vectorstore = Chroma.from_texts(dummy_doc, embedding=embeddings, persist_directory="./chroma_store")
    else:
        vectorstore = FAISS.from_texts(dummy_doc, embedding=embeddings)
    return vectorstore

def add_document_to_store(vs, text: str):
    # Split text into smaller chunks
    chunk_size = 1000  # Adjust based on your needs
    chunk_overlap = 200  # Overlap between chunks for context
    
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    # Add chunks to vectorstore instead of entire document
    vs.add_texts(chunks)

def answer_query(vs, query: str) -> str:
    retriever = vs.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(query)
    return result
