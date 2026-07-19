import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from ingest import clone_repo, load_documents, split_documents
from vectorstore import build_vectorstore
from chatbot import build_chain

# ─── Configuration & Styling ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Repochat AI",
    page_icon=":material/code:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a slight polish
st.markdown("""
<style>
    .stChatInput {
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ─────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.history = []
if "repo_name" not in st.session_state:
    st.session_state.repo_name = ""

# ─── Sidebar Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.title(":material/robot_2: Repochat")
    st.markdown("Chat with any GitHub repository using Groq.")
    st.divider()
    
    st.subheader("1. Setup Repository")
    repo_url = st.text_input(
        "GitHub Repository URL", 
        placeholder="https://github.com/user/repo"
    )
    
    if st.button("Ingest & Build Vector DB", type="primary"):
        if repo_url:
            with st.status("Processing Repository...", expanded=True) as status:
                try:
                    st.write("Cloning repository...")
                    path = clone_repo(repo_url)
                    
                    st.write("Loading documents...")
                    docs = load_documents(path)
                    
                    st.write("Chunking text...")
                    chunks = split_documents(docs)
                    
                    st.write("Building ChromaDB vector store...")
                    vectordb = build_vectorstore(chunks)
                    
                    st.write("Initializing AI chain...")
                    st.session_state.chain = build_chain(vectordb)
                    
                    # Reset chat history and save repo name
                    st.session_state.history = []
                    st.session_state.repo_name = repo_url.split("/")[-1]
                    
                    status.update(label="Repository indexed successfully!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Error indexing repository", state="error")
                    st.error(str(e))
        else:
            st.warning("Please enter a valid GitHub URL.")
            
    st.divider()
    if st.session_state.repo_name:
        st.success(f"**Active Repo:** `{st.session_state.repo_name}`")
    else:
        st.info("No repository loaded.")

# ─── Main Chat Interface ──────────────────────────────────────────────────────
st.title("Repository Assistant")

# Show welcome screen if no repo is loaded
if not st.session_state.chain:
    with st.container(border=True):
        st.markdown("""
        ### :material/waving_hand: Welcome to Repochat! 
        
        This AI assistant is ready to help you explore any codebase. 
        
        **To get started:**
        1. Paste a public GitHub URL in the sidebar on the left.
        2. Click **Ingest & Build Vector DB**.
        3. Once indexing is complete, you can start asking questions about the code right here!
        """)
else:
    # Render chat history
    for msg in st.session_state.history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        avatar = ":material/person:" if role == "user" else ":material/smart_toy:"
        with st.chat_message(role, avatar=avatar):
            st.write(msg.content)
            
    # Chat input
    if question := st.chat_input("Ask something about the codebase..."):
        # Display user message
        with st.chat_message("user", avatar=":material/person:"):
            st.write(question)
            
        # Display assistant response with a spinner
        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            with st.spinner("Analyzing codebase..."):
                try:
                    answer = st.session_state.chain.invoke({
                        "question": question,
                        "history": st.session_state.history,
                    })
                    st.write(answer)
                    
                    # Save to history
                    st.session_state.history.append(HumanMessage(content=question))
                    st.session_state.history.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"Error querying the model: {e}")