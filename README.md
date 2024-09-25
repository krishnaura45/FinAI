<h1 align=center> StockSage: AI Driven Equity Research Tool</h1> 

![image](https://github.com/user-attachments/assets/e4ec6af7-0ecf-49db-bef0-6c906666a755)

## Introduction
Stock Sage is an AI-driven equity research tool designed to automate the research process for equity analysts. By aggregating and analyzing financial news from multiple sources, it allows users to ask specific questions and retrieve concise answers or summaries from news articles. The tool leverages cutting-edge technologies like LangChain, OpenAI API, and Streamlit to streamline the workflow for financial analysts, helping them make data-driven decisions more efficiently.

## Problem Statement
- **Manual Process**: Equity research analysts spend excessive time manually reviewing and summarizing financial news from multiple sources.
- **Fragmented Information**: Key insights are often scattered across various articles, making it difficult to form a comprehensive view.
- **Limitations of Current Tools**: Existing tools, such as ChatGPT, face limitations in handling large documents, multi-source aggregation, and cost-efficiency.
- **Need for Automation**: A tool is required to automate data aggregation, provide concise answers, and streamline the research process, enhancing decision-making accuracy for analysts.

## Literature Review

| **Study**                                             | **Authors**              | **Method**                                                                                     | **Findings**                                                                                   | **Challenges**                                              | **Research Gaps**                                                    | **Improvements in Stock Sage**                              |
|-------------------------------------------------------|--------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------|
| **Automate Strategy Finding with LLM in Quant Investment** | Zhizhuo Kou et al.         | LLMs & multi-agent systems for alpha generation from financial data                            | Outperforms traditional models in trading strategy generation                                  | Computational cost and scalability issues                    | No focus on simplifying equity research tasks                | Automates research by extracting answers from articles       |
| **Revolutionizing Finance with LLMs**                 | M.S. Usha et al.           | LLMs for sentiment analysis and QA                                                              | Enhances sentiment analysis and financial question-answering                                   | Difficulty with ambiguous queries                            | Lacks multi-source aggregation and retrieval                 | Aggregates data from multiple news sources for quick insights |
| **Large Language Models in Finance: A Survey**        | Various Authors           | Survey of LLM applications in financial tasks                                                   | Highlights few-shot/zero-shot learning for financial applications                             | Cost of fine-tuning and handling large documents             | No token optimization or handling large documents            | Reduces API costs by optimizing text chunks for processing    |
| **AI in Equity Research**                             | Recosense Labs            | AI tools like NER and document analyzers                                                        | Automates data extraction, improving accuracy                                                    | Handling unstructured data                                   | No easy-to-use interface for quick querying                  | Provides a simple UI with real-time answers                  |
| **LLMs for Financial Market Forecasting**             | Xiadong Li et al.         | LLMs combined with technical indicators for market forecasting                                 | Improves stock price forecasting accuracy                                                        | Limited by data availability and complexity                  | No integration of multi-source news data for forecasting      | Synthesizes multi-source data for a holistic research view    |

## Objectives
The key objectives of Stock Sage are:
- To automate the process of gathering financial news and research from multiple sources.
- To provide accurate and context-aware answers to user queries based on the aggregated news data.
- To optimize costs by minimizing the amount of text provided to the LLM for analysis, thereby reducing API usage.
- To enhance the efficiency and decision-making process of equity analysts by presenting summarized insights from large volumes of text.

## Working
1. **Data Ingestion**: The tool loads financial news articles using LangChain's document loaders.
2. **Text Processing**: News articles are split into smaller chunks using LangChain’s text splitters to ensure that they remain within LLM token limits.
3. **Embeddings and Storage**: Text chunks are converted into embeddings using OpenAI's embeddings API and stored in a vector database (FAISS) for quick retrieval.
4. **Query Processing**: When a user asks a question, the system retrieves relevant text chunks from the vector database and formulates an LLM prompt.
5. **Answer Generation**: The LLM generates an answer, which is returned to the user along with the source of the information.

## Summary
**StockSage** is one of the cutting-edge news research tools designed to simplify information retrieval from the stock market and financial domain. With this analyzer, users can effortlessly input article URLs and ask questions to receive relevant insights, making it an invaluable asset for investors, analysts, and financial enthusiasts.

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

## Conclusion
Stock Sage can successfully automate the equity research process by allowing analysts to input URLs from financial news sources and retrieve answers quickly and efficiently. This will not only save time but also enhance the accuracy of their research by ensuring that all relevant sources are considered. The tool also provides summarized insights from large documents, improving the decision-making process for equity analysts.

## Future Scope
- **Integration with Real-time Data**: Extend the tool to handle real-time stock data and integrate financial reports.
- **Advanced Summarization**: Implement advanced summarization techniques using fine-tuned models for industry-specific insights.
- **Multi-Language Support**: Enable multi-language support for global equity research.
- **Custom Model Fine-tuning**: Explore fine-tuning custom LLM models specifically for financial domains to improve domain-specific accuracy.

## References
1. **Kou, Zhizhuo, Alan, and Shen.** "Automate Strategy Finding with LLM in Quant Investment." *arXiv preprint arXiv:2409.06289*, 2024.
2. **Usha, M.S., and Kirange, D.K.** "Revolutionizing Finance with LLMs: An Overview of Applications and Insights." *arXiv preprint arXiv:2401.11641*, 2024.
3. **Touvron, Hugo, and Leclerc, G.** "Large Language Models in Finance: A Survey." *arXiv preprint arXiv:2311.10723*, 2023.
4. **Recosense Labs.** "AI in Equity Research." *Recosense Labs Publication*, 2024.
5. **Li, Xiadong, and Zhang, C.** "LLMs for Financial Market Forecasting." *arXiv preprint arXiv:2311.10723*, 2023.

## Tech Stacks Involved
- **LangChain**: Used for document loading, chunking, and integrating the LLM.
- **OpenAI API**: For embedding text chunks and generating responses.
- **Streamlit**: To build a user-friendly web interface for querying and displaying results.
- **FAISS (Facebook AI Similarity Search)**: For efficient vector-based search.
- **Python**: For backend development and API integration.
- **Git**: For version control.

