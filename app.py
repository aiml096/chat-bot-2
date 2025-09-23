import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from pinecone import Pinecone
from dotenv import load_dotenv

# ----------------------------
# LOAD ENV
# ----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pdf-hf-index"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------------------
# FASTAPI APP
# ----------------------------
app = FastAPI(title="Company RAG Chatbot with Pinecone")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------------------
# PINECONE & EMBEDDINGS
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ----------------------------
# RAG QA CHAIN WITH LLM
# ----------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# ----------------------------
# QUERY ENDPOINT
# ----------------------------
@app.get("/chat")
def chat(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        answer = qa_chain.run(query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
