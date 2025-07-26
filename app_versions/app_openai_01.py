# first version
import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# # To check api key is loaded or not
# print(os.getenv("OPENAI_API_KEY"))

# # Checking LLM from openai API
# try:
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
#     print("Test output:", llm.predict("Hello, what is stock market?"))
# except Exception as e:
#     print("LLM Error:", e)

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url= st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss-store-openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading >>> Started >>> ✅✅✅")
    data = loader.load()
    print(len(data))
    print(data)

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter >>> Started >>> ✅✅✅")
    docs = text_splitter.split_documents(data)
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore_openai = pickle.load(f)
    else:
        # create embeddings and save it to FAISS index
        embeddings = OpenAIEmbeddings()
        
        # Process and save the Faiss index into a pickle file
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
    main_placeholder.text("Embedding Vector Creation >>> ✅✅✅")
    time.sleep(2)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)