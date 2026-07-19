#  Retriever + Groq chain, built with LCEL
# This is where retrieval and generation actually connect. 
# Given a question, the retriever pulls the most relevant chunks from Chroma, 
# they get formatted into the prompt alongside the conversation history, 
# and Groq's LLM generates an answer grounded in that context.

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = (
    "You are an expert code reviewer and software engineer. "
    "You are helping a developer understand a GitHub repository. "
    "You MUST answer ONLY from the code and files in the Context section below. "
    "If the Context is empty or does not contain the answer, "
    "respond ONLY with: 'I cannot find that information in the repository.' "
    "NEVER answer from your general training knowledge. "
    "NEVER answer questions that are unrelated to the repository. \n\n"

    "When answering: \n"
    "1. Be specific — reference actual function names, file paths, "
    "and variable names from the context. \n"
    "2. If the question is about HOW something works, walk through "
    "the logic step by step. \n"
    "3. If the question is about WHERE something is, give the exact "
    "file path and line context. \n"
    "4. If the question is about WHY a design decision was made, "
    "reason from the code you can see. \n\n"

    "Never make up function names, file names, or behaviour "
    "that you cannot see in the context. \n\n"

    "Always wrap code snippets in triple backticks with the language "
    "name e.g. ```python ... ``` \n\n"

    "Context:\n{context}"
)

def build_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs = {"k": 5})
    llm = ChatGroq(model = "openai/gpt-oss-120b", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('source')}]\n{d.page_content}" for d in docs
        )
    
    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
    
