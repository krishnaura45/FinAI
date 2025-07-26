import os
import pickle
import time
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
from langchain.llms import LlamaCpp
from dotenv import load_dotenv
from llama_cpp import Llama

# Load environment variables
load_dotenv()

# --- Settings ---
MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"  # Update if your path differs
file_path = "faiss-store-hf.pkl"

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Sage - Local", layout="wide")
st.title("Stock Sage (🔒 Local Mode)")
st.sidebar.title("📄 Enter News URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("🔍 Process URLs")
main_placeholder = st.empty()

# --- Load LLM ---
st.text("[STEP 0] Loading LLM locally...")
llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=1024,
    n_batch=512,
    f16_kv=True,
    verbose=True
)

# --- Handle URL processing ---
if process_url_clicked:
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    st.success(f"[STEP 1] Loaded {len(data)} documents ✅")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    st.success(f"[STEP 2] Split into {len(docs)} chunks ✅")

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

    st.success("[STEP 3] Vectorstore created and saved ✅")

# --- Handle Querying ---
query = main_placeholder.text_input("Ask a question about the articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            if result.get("sources"):
                st.subheader("Sources")
                for source in result["sources"].split("\n"):
                    st.write(source)