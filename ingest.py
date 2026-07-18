import os
import shutil
from git import Repo

# Clone GitHub Repo
def clone_rep(repo_url :str, clone_dir :str = "cloned/current")-> str:
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    Repo.clone_from(repo_url, clone_dir, depth =1)
    return clone_dir



# Loading and Chunking
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

IGNORE_EXT = {".png", ".jpg", ".gif", ".svg", ".lock", ".woff", ".ttf", ".ico"}

def load_documents(repo_path: str):
    loader = DirectoryLoader(
        repo_path,
        glob="**/*",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        silent_errors=True,
    )
    docs = loader.load()
    return [d for d in docs
            if not any(d.metadata["source"].endswith(e) for e in IGNORE_EXT)]

# Map file extension to Language enum
LANGUAGE_MAP = {
    ".py":  Language.PYTHON,
    ".js":  Language.JS,
    ".ts":  Language.JS,       # TypeScript, close enough
    ".md":  Language.MARKDOWN,
    ".html": Language.HTML,
    ".cpp": Language.CPP,
    ".java": Language.JAVA,
}

def split_documents(docs):
    all_chunks = []

    # Group docs by their file extension
    from collections import defaultdict
    groups = defaultdict(list)

    for doc in docs:
        # Get file extension e.g. ".py" from "repo/src/app.py"
        ext = os.path.splitext(doc.metadata["source"])[1].lower()
        groups[ext].append(doc)

    # Split each group with the right language splitter
    for ext, group_docs in groups.items():
        language = LANGUAGE_MAP.get(ext, None)

        if language:
            # Use language-aware splitter
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=1000,
                chunk_overlap=150
            )
        else:
            # Unknown extension -> plain text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

        chunks = splitter.split_documents(group_docs)
        all_chunks.extend(chunks)

    return all_chunks