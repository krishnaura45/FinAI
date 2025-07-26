import os
import pickle
import time
import streamlit as st
from langchain.llms import GPT4All
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

MODEL_PATH = "models/gpt4all-lora-quantized.bin"
VECTORSTORE_PATH = "faiss-store-hf.pkl"

st.set_page_config(page_title="Stock Sage - GPT4All", layout="wide")
st.title("Stock Sage (🤖 GPT4All Local Mode)")
st.sidebar.title("🔗 Enter URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_button = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# Load LLM
llm = GPT4All(model=MODEL_PATH, verbose=True)

if process_button:
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    st.success(f"✅ Loaded {len(data)} documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(data)

    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(VECTORSTORE_PATH, "wb") as f:
            pickle.dump(vectorstore, f)

    st.success("✅ Embeddings vectorstore created")

query = main_placeholder.text_input("🔍 Ask your question:")
if query:
    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, "rb") as f:
            vectorstore = pickle.load(f)
            qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = qa_chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])
            if result.get("sources"):
                st.subheader("Sources:")
                for src in result["sources"].split("\n"):
                    st.write(src)