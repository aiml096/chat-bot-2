from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # Only for example if using local LLM, replace if needed
from dotenv import load_dotenv
app = FastAPI(title="Company RAG Chatbot")
load_dotenv()
# Allow frontend JS to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_methods=["*"],
    allow_headers=["*"]
)

CHROMA_DB_DIR = "./docs/chroma_db"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Chroma retriever
retriever = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
).as_retriever(search_kwargs={"k": 4})  # <-- Important: convert to retriever

# Create RAG QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGroq(model_name="llama-3.1-8b-instant", temperature=0),  # Replace with Groq LLM if you have it
    retriever=retriever,
    return_source_documents=False
)

@app.get("/chat")
def chat(query: str):
    try:
        answer = qa_chain.run(query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
