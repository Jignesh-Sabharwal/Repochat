from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_vectorstore(chunks, persist_dir: str = "chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectordb


def load_vectorstore(persist_dir: str = "chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name = EMBED_MODEL)
    return Chroma(persist_directory = persist_dir, embedding_function = embeddings)
    