import os
import time
import pickle
import streamlit as st
import google.generativeai as genai
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class GeminiLLM(LLM):
    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        model = genai.GenerativeModel("gemini-1.5-flash")
        try:
            start_time = time.time()
            response = model.generate_content(prompt)
            print(f"[Gemini] Response received in {round(time.time() - start_time, 2)} sec")
            return response.text
        except Exception as e:
            print(f"[Gemini ERROR] {e}")
            return "Gemini failed to respond."

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

st.title("Debug News Research Tool 🧪")

# Minimal single working URL
urls = ["https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html"]
file_path = "test/faiss-store-debug.pkl"
main_placeholder = st.empty()
llm = GeminiLLM()

if st.sidebar.button("Run Debug Pipeline"):
    st.sidebar.write("Step 1: Loading URLs...")
    start_time = time.time()
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    print(f"[STEP 1] Loaded {len(data)} documents in {round(time.time() - start_time, 2)} sec")

    st.sidebar.write("Step 2: Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, separators=["\n\n", "\n", ".", " "])
    docs = text_splitter.split_documents(data)
    print(f"[STEP 2] Split into {len(docs)} chunks")

    st.sidebar.write("Step 3: Creating embeddings...")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            print("[STEP 3] Loaded embeddings from file.")
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
            print("[STEP 3] New vectorstore created and saved.")

    st.sidebar.write("✅ Setup Complete")

query = main_placeholder.text_input("Ask a question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            start_query_time = time.time()
            result = chain({"question": query}, return_only_outputs=True)
            print(f"[STEP 4] Query handled in {round(time.time() - start_query_time, 2)} sec")

            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for s in sources.split("\n"):
                    st.write(s)
