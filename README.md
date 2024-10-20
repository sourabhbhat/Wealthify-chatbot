# Wealthify-chatbot

# Wealthify: Ask Me Anything on Mutual Funds

Wealthify Chat is an AI-powered chatbot which inturn is an extended feature of a bigger project Wealthify, that recommends mutual funds,ETFs and Bonds. Built using `Streamlit`, `LangChain`, `ChatGroq`, and `Google Generative AI Embeddings`, this app allows users to ask custom questions or use provided sample questions. The app processes and retrieves context from financial documents to give accurate, insightful answers.

## Features

- **Interactive Question-Answering**: Ask questions about mutual funds and get answers based on financial documents.
- **Sample Questions**: Three sample questions are provided for easy exploration of the app.
- **Automatic Document Embedding**: On app load, the mutual fund documents are automatically embedded into a vector store for fast search and retrieval.
- **Contextual Retrieval**: Uses vector search to retrieve the most relevant sections of the documents and provide accurate answers.

## Technologies Used

- **Streamlit**: For building the interactive user interface.
- **LangChain**: For creating document retrieval and language model chains.
- **ChatGroq**: For interacting with the Gemma-7b-it model.
- **Google Generative AI Embeddings**: For generating vector embeddings of the mutual fund documents.
- **FAISS**: For efficient vector search and retrieval.
- **PyPDF**: For loading and processing PDF documents.
- **Python**: The app is built using Python, making it easy to run and extend.

## How It Works

1. **Document Embedding**: On app load, mutual fund PDFs from the `mf_docs` folder are ingested, split into smaller chunks, and embedded as vectors.
2. **Asking Questions**: Users can either input their own question or click on one of the sample question buttons.
3. **Contextual Answering**: Based on the user's input, the app retrieves the most relevant document chunks, uses the Gemma model for answering, and displays the result.
4. **Similarity Search**: The app allows users to explore more related document sections via a document similarity search.

## Setup Instructions

### Prerequisites

- Python 3.x
- [GROQ API Key](https://www.groq.com/)
- [Google API Key](https://console.cloud.google.com/apis/credentials) for Generative AI Embeddings

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/wealthify-chatbot.git
   cd wealthify-chatbot
