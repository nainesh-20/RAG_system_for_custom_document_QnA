# Retrieval Augmented Generation (RAG) System for Custom Document Q&A

## Project Overview

This project implements a **Retrieval Augmented Generation (RAG)** pipeline designed to enable accurate and context-aware question answering over custom document sets. By combining the power of Large Language Models (LLMs) with efficient information retrieval from a vector database, the system can provide answers grounded in your specific data, overcoming the limitations of general-purpose LLMs on domain-specific or private knowledge.

The project demonstrates the key steps involved in building a RAG application using modern **Langchain** components, including document loading, splitting, embedding, vector storage (both local and cloud), and constructing the query pipeline.

## Problem Addressed

Large Language Models are powerful but lack knowledge about specific, private, or recently updated information that wasn't part of their training data. Directly asking an LLM about your internal documents would result in inaccurate or fabricated answers. This project solves that by:

1.  Storing your documents' knowledge in a searchable format (vector database).
2.  Retrieving relevant snippets from your documents based on a user's query.
3.  Providing these retrieved snippets to the LLM as context, enabling it to generate accurate answers based *only* on the provided information.

## Features

* **Document Ingestion:** Load documents from local files (e.g., PDFs) using Langchain Document Loaders.
* **Text Processing:** Split documents into optimized chunks using a token-based Recursive Character Text Splitter (`tiktoken`).
* **Embedding:** Generate semantic embeddings for document chunks using OpenAI Embeddings.
* **Vector Storage:** Support for both:
    * **FAISS:** A local, in-memory/disk vector database for prototyping and testing.
    * **Pinecone:** A scalable cloud vector database for persistent storage and production-like environments.
* **Semantic Search:** Perform efficient similarity search against the vector database to retrieve context relevant to a user query.
* **RAG Query Pipeline:** Construct a RAG chain using modern Langchain Expression Language (LCEL) utilities (`create_retrieval_chain`, `create_stuff_documents_chain`) to pass retrieved context and the user query to an LLM.
* **Context-Aware Answering:** LLM generates answers based *only* on the provided document context.
* **Source Citation:** Prompt the LLM to cite the source (document name, page number) from the metadata of the retrieved chunks.
* **Out-of-Scope Handling:** Instruct the LLM to decline answering questions if relevant information is not found in the retrieved context.
* **Modular Design:** Code structured to demonstrate individual steps of the RAG pipeline.

## Technologies Used

* **Python:** Core programming language.
* **Langchain:** Framework for developing applications powered by language models.
    * Langchain Expression Language (LCEL)
    * Document Loaders (`langchain-community`)
    * Text Splitters
    * Embeddings (`langchain-openai`)
    * Vector Stores (`langchain-community`, `langchain-pinecone`)
    * Chains & Runnables
    * Chat Models (`langchain-openai`)
    * Memory (`langchain`)
* **OpenAI API:**
    * Large Language Models (e.g., `gpt-3.5-turbo`) for generation.
    * Embedding Models (e.g., `text-embedding-ada-002`) for vectorization.
* **Pinecone:** Cloud-based vector database service.
* **FAISS:** Local library for efficient similarity search and clustering of dense vectors.
* **`tiktoken`:** OpenAI's fast BPE token encoder for token counting.
* **`python-dotenv`:** To load environment variables from a `.env` file.
* **`pandas` & `matplotlib`:** For optional data visualization (e.g., chunk size distribution).
* **`pypdf`:** Dependency for PDF loading.
* **`wikipedia`:** Dependency for Wikipedia loading example.

## Getting Started

Follow these instructions to set up and run the project.

### Prerequisites

* Python 3.8+
* An OpenAI API Key
* (Optional) A Pinecone Account and API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```
    (Replace `<repository_url>` and `<repository_folder>` with your project's actual details)

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    * On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    * On Windows:
        ```bash
        .venv\Scripts\activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (You'll need a `requirements.txt` file listing all the libraries used: `langchain`, `langchain-openai`, `langchain-community`, `langchain-pinecone`, `pinecone-client`, `python-dotenv`, `pypdf`, `tiktoken`, `pandas`, `matplotlib`, `wikipedia`). If you don't have one, create it manually or generate it: `pip freeze > requirements.txt` after installing everything).

### Environment Variables

1.  Create a file named `.env` in the root directory of the project.
2.  Add your API keys and Pinecone environment details to this file:
    ```dotenv
    OPENAI_API_KEY='your-openai-api-key'
    PINECONE_API_KEY='your-pinecone-api-key'
    PINECONE_ENVIRONMENT='your-pinecone-environment' # e.g., 'us-east-1-aws'
    PINECONE_INDEX_NAME='your-index-name' # e.g., 'langchain-index'
    ```
    Replace the placeholder values with your actual keys and details.

### Running the Notebook

The project is structured as a Jupyter Notebook (or executed cell-by-cell in environments like VS Code). The steps of the RAG pipeline are implemented in sequence in different cells.

1.  **Open the main notebook file** (e.g., `main.ipynb`) in your preferred environment (Jupyter, VS Code, etc.).
2.  **Run the cells sequentially.** Follow the comments and print statements in each cell.
    * The notebook starts with loading environment variables and basic component initialization.
    * It then proceeds through the **Data Ingestion** steps: Loading documents (ensure your `Docs/` directory exists and contains PDFs, update paths as needed), splitting, embedding, and setting up/populating the vector stores (FAISS and Pinecone).
    * Finally, it moves to the **Querying (RAG)** steps, demonstrating semantic search and the RAG query pipeline using both FAISS and Pinecone.
    * Pay attention to instructions for specific cells (like adjusting paths or ensuring previous cells ran successfully before proceeding).

## Project Structure (Example based on discussion)

├── .venv/              # Python virtual environment
├── .env                # Environment variables (API keys, etc.)
├── Docs/               # Directory to place your PDF documents
│   └── your_document.pdf
├── requirements.txt    # Project dependencies
├── rag_notebook.ipynb  # Main Jupyter Notebook with project code
└── README.md           # Project README file

## Future Enhancements

* Implement **Conversational RAG** to support multi-turn dialogue while maintaining context from chat history and retrieved documents (using `create_history_aware_retriever` and memory).
* Build a simple **User Interface** (e.g., using Streamlit or Gradio) to interact with the chatbot.
* Add support for loading documents from **more data sources** (databases, websites, etc.).
* Implement **evaluation metrics** to assess retrieval and generation quality.

## License

*MIT, Apache 2.0*

## Contact

*naineshrathod3000@gmail.com*
*LinkedIn: https://www.linkedin.com/in/nainesh-rathod/*

---
