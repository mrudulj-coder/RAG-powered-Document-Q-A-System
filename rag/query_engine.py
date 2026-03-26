"""RAG query engine using LangChain and Google Gemini."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from rag.vector_store import get_retriever
from config import GOOGLE_API_KEY, LLM_MODEL, TEMPERATURE


def _get_llm() -> ChatGoogleGenerativeAI:
    """Initialize Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
        convert_system_message_to_human=True,
    )


# ── Prompt: Reformulate question using chat history ──────────────────

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question which might "
            "reference context in the chat history, formulate a standalone "
            "question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and "
            "otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ── Prompt: Answer grounded in retrieved context ─────────────────────

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant that answers questions based "
            "on the provided document context. Follow these rules:\n\n"
            "1. Answer ONLY based on the provided context below.\n"
            "2. If the context doesn't contain enough information, say so clearly.\n"
            "3. Cite which document and page your answer comes from.\n"
            "4. Be concise but thorough.\n"
            "5. Use bullet points or numbered lists for clarity when appropriate.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def create_rag_chain():
    """
    Build a conversational RAG chain with history-aware retrieval.

    The chain:
    1. Reformulates the user question using chat history for context
    2. Retrieves relevant document chunks from ChromaDB
    3. Generates a grounded answer using Gemini with source citations
    """
    llm = _get_llm()
    retriever = get_retriever()

    # Step 1: History-aware retriever (reformulates question)
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_PROMPT
    )

    # Step 2: QA chain that stuffs docs into prompt
    qa_chain = create_stuff_documents_chain(llm, QA_PROMPT)

    # Step 3: Full retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain


def query(question: str, chat_history: list) -> dict:
    """
    Query the RAG system with conversation history.

    Args:
        question: The user's question.
        chat_history: List of (human_msg, ai_msg) tuples.

    Returns:
        dict with 'answer' and 'context' (source documents).
    """
    rag_chain = create_rag_chain()

    # Convert chat history to LangChain message format
    history_messages = []
    for human_msg, ai_msg in chat_history:
        history_messages.append(HumanMessage(content=human_msg))
        history_messages.append(AIMessage(content=ai_msg))

    result = rag_chain.invoke(
        {"input": question, "chat_history": history_messages}
    )

    return {
        "answer": result["answer"],
        "context": result.get("context", []),
    }
