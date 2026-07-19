# 🤖 Repochat AI

Chat with any public GitHub repository using natural language. Repochat AI clones a repo, indexes its source files into a vector database, and lets you ask questions about the codebase — architecture, function behavior, file locations, and design decisions — through a Streamlit chat interface powered by Groq.

## Features

- **Clone & Ingest**: Paste any public GitHub URL and Repochat will clone and process it automatically.
- **Language-Aware Chunking**: Source files are split using language-specific splitters (Python, JS/TS, Markdown, HTML, C++, Java) for more coherent context chunks.
- **Semantic Search**: Uses HuggingFace sentence-transformer embeddings (`all-MiniLM-L6-v2`) with a local ChromaDB vector store.
- **Grounded Answers**: The assistant is instructed to answer strictly from retrieved repository context — no hallucinated APIs or made-up file names.
- **Conversational Memory**: Maintains chat history for follow-up questions within a session.
- **Fast Inference**: Powered by Groq's `openai/gpt-oss-120b` model via `langchain-groq`.

## Tech Stack

| Component        | Technology                                  |
|-------------------|----------------------------------------------|
| UI                | [Streamlit](https://streamlit.io)             |
| Orchestration     | [LangChain](https://www.langchain.com) (LCEL) |
| LLM               | Groq (`openai/gpt-oss-120b`)                  |
| Embeddings        | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store      | ChromaDB                                      |
| Repo Cloning      | GitPython                                     |

## Project Structure

```
.
├── app.py            # Streamlit UI and session state management
├── ingest.py          # Repo cloning, document loading, and chunking
├── vectorstore.py     # Embedding + Chroma vector store creation/loading
├── chatbot.py          # Retrieval-augmented generation chain (LCEL)
└── requirements.txt    # Python dependencies
```

## Prerequisites

- Python 3.9+
- Git installed and available on your `PATH`
- A [Groq API key](https://console.groq.com/keys)

## Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/Jignesh-Sabharwal/Repochat.git
   cd Repochat
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root with your Groq API key:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

1. **Run the app**

   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to the local URL Streamlit provides (typically `http://localhost:8501`).

3. **Index a repository**
   - Paste a public GitHub repository URL into the sidebar.
   - Click **Ingest & Build Vector DB**.
   - Wait for cloning, chunking, and indexing to complete.

4. **Start chatting**
   - Ask questions like:
     - "Where is the authentication logic handled?"
     - "How does the `build_chain` function work?"
     - "Why was ChromaDB chosen over another vector store?"

## How It Works

1. **Ingest** (`ingest.py`): Clones the target repo with a shallow (`depth=1`) clone, loads text files (skipping binaries like images and fonts), and splits them into chunks using language-aware splitters where available.
2. **Vectorize** (`vectorstore.py`): Each chunk is embedded using a MiniLM sentence-transformer model and stored in a local ChromaDB instance.
3. **Chat** (`chatbot.py`): For every user question, the top-k most relevant chunks are retrieved from Chroma, formatted into a system prompt with strict grounding instructions, and passed to Groq's LLM alongside conversation history to generate a context-aware answer.
4. **UI** (`app.py`): Streamlit manages session state, renders the chat interface, and orchestrates the ingest → vectorize → chat pipeline.

## Notes & Limitations

- Only **public** GitHub repositories are supported out of the box (no authentication handling for private repos).
- Binary and asset files (images, fonts, lockfiles, etc.) are excluded from ingestion.
- The vector store is persisted locally to a `chroma_db` directory; re-ingesting a repo will rebuild it.
- Answers are intentionally restricted to retrieved context — if the answer isn't in the indexed repo, the assistant will say so rather than guessing.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for bug fixes, new features, or improvements.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
