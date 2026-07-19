# Repochat AI 🤖

An interactive, Retrieval-Augmented Generation (RAG) chatbot that allows you to seamlessly converse with any public GitHub repository. 

Built with **LangChain**, **ChromaDB**, and **Streamlit**, Repochat clones a repository on the fly, embeds its codebase locally, and uses the ultra-fast **Groq** API to answer your questions strictly based on the actual code!

## ✨ Features

- **On-the-Fly Ingestion:** Paste any GitHub URL to instantly clone, chunk, and index the repository.
- **Local Embeddings:** Uses HuggingFace's `sentence-transformers` to generate fast, local, and private embeddings.
- **Strict Anti-Hallucination:** System prompts are strictly engineered to ensure the AI *only* answers using the provided codebase context.
- **Premium UI:** A highly polished, custom-themed Streamlit interface with sidebar navigation and material icons.
- **Blazing Fast AI:** Powered by the Groq API for near-instantaneous LLM inference.

## 🏗️ Architecture

- **Frontend:** Streamlit (`app.py`)
- **Orchestration:** LangChain (`LCEL`)
- **Vector Database:** ChromaDB
- **Embeddings Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** Groq API

## 🚀 Getting Started

### 1. Clone this repository
```bash
git clone https://github.com/YourUsername/Repochat.git
cd Repochat
```

### 2. Set up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY="your_groq_api_key_here"
```

### 5. Run the Application
```bash
streamlit run app.py
```

## 📁 Project Structure

- `app.py`: The main Streamlit user interface and application entry point.
- `chatbot.py`: Contains the LangChain LCEL implementation, combining the retriever, history, and Groq LLM.
- `ingest.py`: Handles cloning the GitHub repository, loading text files, and splitting them into chunks.
- `vectorstore.py`: Manages the creation and loading of the local ChromaDB vector database.
- `.streamlit/config.toml`: Contains the custom dark mode theme configuration and terminal spam fixes.
