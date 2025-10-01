import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from pinecone import Pinecone, AsyncPinecone
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# ----------------------------
# LOAD ENV
# ----------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pdf-hf-index"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Company RAG Chatbot with Pinecone")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------------------
# GLOBALS INITIALIZED AT STARTUP
# ----------------------------
class AppState:
    vectorstore = None
    retriever = None
    qa_chain = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Async Pinecone client and vectorstore
    pc = AsyncPinecone(api_key=PINECONE_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, client=pc)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    # Store shared objects on state
    state.vectorstore = vectorstore
    state.retriever = retriever
    state.qa_chain = qa_chain
    yield
    # Optional: Clean up resources if needed

# Re-register the lifespan event
app.router.lifespan_context = lifespan

# ----------------------------
# QUERY ENDPOINT (ASYNC)
# ----------------------------
@app.get("/chat")
async def chat(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        # If qa_chain is sync (depends on langchain_groq implementation), wrap in thread pool:
        from starlette.concurrency import run_in_threadpool
        answer = await run_in_threadpool(state.qa_chain.run, query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
