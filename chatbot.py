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

    # Tell it HOW to answer, not just what to do
    "When answering: \n"
    "1. Be specific — reference actual function names, file paths, "
    "and variable names from the context. \n"
    "2. If the question is about HOW something works, walk through "
    "the logic step by step. \n"
    "3. If the question is about WHERE something is, give the exact "
    "file path and line context. \n"
    "4. If the question is about WHY a design decision was made, "
    "reason from the code you can see. \n"

    # Prevent hallucination
    "If the retrieved context does not contain enough information "
    "to answer confidently, say exactly what you DO know from the "
    "context and what you are unsure about. "
    "Never make up function names, file names, or behaviour "
    "that you cannot see in the context. \n\n"

    # Code formatting
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
    
