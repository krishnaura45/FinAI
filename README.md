<h1 align=center> FinAI: LLM-based Equity Research Engine</h1> 

![Tool UI](https://github.com/user-attachments/assets/5c066a63-4df4-4bbb-8f3c-852b7aeed32a)

---

## Table of Contents

* [Introduction](#introduction)
* [Problem Statement](#problem-statement)
* [Working (Pipeline Stages)](#working-pipeline-stages)
* [Results](#results)
* [Files & Structure](#files--structure)
* [Installation](#installation)
* [Tech Stack](#tech-stack)
* [References](#references)
* [Future Scope](#future-scope)
* [Contributing](#contributing)
* [Thanks for Visiting!](#thanks-for-visiting)

---

## Introduction 📄

**FinAI** is a Retrieval-Augmented Generation (RAG) based equity news analyzer that simplifies information retrieval for investors, analysts, and financial researchers. Built using LangChain, OpenAI, Gemini, FAISS, and local LLM backends, it allows users to input article URLs and query them in natural language.

---

## Problem Statement 🚫

* Equity research is **manual, fragmented, and time-consuming**.
* Analysts must manually browse multiple sources and interpret insights.
* LLMs like ChatGPT alone cannot handle **multi-source**, **large documents**, or **real-time querying efficiently**.
* Need for a tool that can ingest articles, process them intelligently, and provide **accurate, real-time answers**.

---

## Working (Pipeline Stages) ⚙️

1. **Data Ingestion**

   * News article URLs are fetched using `SeleniumURLLoader`.

2. **Text Splitting**

   * Articles are chunked using LangChain's `RecursiveCharacterTextSplitter` to fit LLM token limits.

3. **Embeddings & Vector Store**

   * Embeddings created via `OpenAI` or `HuggingFace` models (like `all-MiniLM-L6-v2`).
   * FAISS stores and retrieves similar content based on queries.

4. **Querying via LLMs**

   * User queries are answered using OpenAI/Gemini/GPT4All/LLama-2 LLMs via `RetrievalQAWithSourcesChain`.
   * Local models (`llama-cpp-python`, `gpt4all`) ensure offline support.

5. **Answer + Source Display**

   * Source-linked responses shown via Streamlit.

---

## Results 🔄

* Tool providing real-time updates on each stage’s progress
![image](https://github.com/user-attachments/assets/097f5105-ea4c-404d-9cfe-fb24c74464d9)

* Tool providing exact accurate answer on straightforward (direct) queries
![image](https://github.com/user-attachments/assets/7f9f03c6-6b0f-4d25-ab01-5f384a9fe960)

* Customized same pipeline using both **online APIs** (OpenAI, Gemini) and **offline models** (LLaMA, GPT4All).

* Benchmark Table

  | **Backend**            | **Avg Latency** | **QA Relevance** | **Token Cost** | **Use-Case Fit**                            |
  |---------------------- | --------------- | ---------------- | -------------- | ------------------------------------------- |
  | OpenAI (gpt-3.5-turbo) | \~2.1s          | 96.4%            | High (Paid)    | Best for fast, high-quality responses     |
  | Gemini Pro             | \~2.8s          | 92.1%            | Free (limited) | Good fallback; prone to hallucination     |
  | Local LLaMA (7B)       | \~5.3s          | 93.2%            | None           | Reliable offline QA; requires setup       |
  | GPT4All (q4\_0)        | \~7.2s          | 86.5%            | None           | Works offline; lower accuracy in deep QA |


---

## Files & Structure 📁

- `app_versions/`: Contains different Streamlit app versions based on LLMs — OpenAI, Gemini, GPT4All, and LLaMA.
- `data_files/`: Includes sample article text files and URL lists used during experimentation.
- `notebooks/`: Jupyter notebooks demonstrating individual components of the RAG pipeline (e.g., vector store testing, embeddings, chunking).
- `test/`: Debugging and testing scripts for Gemini and LLaMA-based app flows.
- `.env`: Stores environment variables like API keys for OpenAI and Gemini.
- `faiss-store-hf.pkl`: Vector store generated using HuggingFace embeddings.
- `faiss-store-openai.pkl`: Vector store generated using OpenAI embeddings.
- `vector-index.pkl`: Sample vector index created using notebook for FAISS validation.
- `main.py`: Primary file containing final UI code after experimentation.
- `requirements.txt`: Python dependencies required for running the project.
- `README.md`: Documentation and usage guide for the project.
- `models/`: 🔐 Not uploaded — should contain downloaded local LLMs (refer to Installation)

---

## Installation 🚧

1. **Clone the repository**

```bash
git clone https://github.com/your-username/FinAI.git
cd FinAI
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up API keys**
   Create a `.env` file just like the reference being provided and add:

```
OPENAI_API_KEY=your-key-here
GOOGLE_API_KEY=your-google-studio-api-key
```

* OpenAI Key: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
* Gemini Key: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (enable Gemini API on Google Cloud Console)

4. **For local LLM usage**

* Download `.gguf` models from [LLaMA HF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) or [GPT4All](https://gpt4all.io/index.html).
* Create a `models/` directory and place them inside.
* Update `model_path` in corresponding app files (e.g. `app_local_llama.py`).

5. **Run the tool**

```bash
streamlit run main.py
```

---

## Tech Stack 🚀

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.0.284-yellow?logo=langchain)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-ff4b4b?logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-API-04a6d4?logo=openai)
![Google Gemini](https://img.shields.io/badge/Gemini-API-34a853?logo=google)
![FAISS](https://img.shields.io/badge/FAISS-1.7.4-4b8bbe?logo=facebook)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-fcc624?logo=huggingface)
![LLaMA](https://img.shields.io/badge/LLaMA-2.7B-green?logo=meta)
![GPT4All](https://img.shields.io/badge/GPT4All-Local%20LLM-purple?logo=nvidia)

* **LangChain**: Orchestration of RAG pipeline
* **Streamlit**: Interactive web interface
* **OpenAI & Gemini APIs**: Cloud-based LLMs
* **LLaMA / GPT4All**: Local LLMs
* **HuggingFace Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
* **FAISS**: Vector similarity search and store
* **Python + Selenium**: Document scraping + automation

---

## References 📚

* [OpenAI Platform](https://platform.openai.com/)
* [Gemini API Studio](https://aistudio.google.com/app/apikey)
* [Google Cloud Console](https://console.cloud.google.com/)
* [HuggingFace Embedding Model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* [LLaMA GGUF HF Models](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
* [GPT4All Local Models](https://gpt4all.io/index.html)
* [Other Models](https://huggingface.co/nomic-ai/models)
* [Streamlit Docs](https://docs.streamlit.io)

---

## Future Scope 🔮

* Real-time financial API integration (e.g. stock prices, reports)
* LLM-based summarization for multi-source insights
* Domain-tuned custom LLMs for financial jargon
* Globalization support via multi-language ingestion

---

## Contributing 🤝

We welcome contributions! Feel free to:

* Fork the repo
* Create a new branch
* Submit PR with changes or improvements

---

## Thanks for Visiting 😊!

We hope **FinAI** helps you gain actionable insights with less effort. If you like it, give the repo a ⭐ and feel free to reach out for suggestions or ideas!

---
<!--
## Summary
**FinAI** is one of the cutting-edge news research tools designed to simplify information retrieval from the stock market and financial domain. With this analyzer, users can effortlessly input article URLs and ask questions to receive relevant insights, making it an invaluable asset for investors, analysts, and financial enthusiasts.

### Features
- **Effortless URL Loading**
  - Input Flexibility: Load URLs directly or upload text files containing multiple URLs for batch processing.
  - Content Fetching: Automatically fetch article content using LangChain's UnstructuredURL Loader for seamless integration.
    
- **Advanced Information Retrieval**
  - Embedding Construction: Construct high-quality embedding vectors using OpenAI's state-of-the-art embeddings.
  - Similarity Search: Leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information.

- **Interactive Querying**
  - Natural Language Interaction: Interact with advanced Language Learning Models (LLMs) like ChatGPT by inputting queries.
  - Insightful Responses: Receive detailed answers along with source URLs, ensuring transparency and reliability of information.
-->
