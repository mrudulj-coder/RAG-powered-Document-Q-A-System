"""RAG query engine using LangChain Expression Language and Google Gemini."""

import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.vector_store import get_retriever
from config import GOOGLE_API_KEY, LLM_MODEL, TEMPERATURE

logger = logging.getLogger(__name__)

# ── Retry Settings ───────────────────────────────────────────────────
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 10  # seconds


def _get_llm() -> ChatGoogleGenerativeAI:
    """Initialize Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
    )


def _invoke_with_retry(llm, messages, max_retries=MAX_RETRIES):
    """Invoke LLM with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"Rate limited (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {delay}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"⏳ Gemini API rate limit exceeded after {max_retries} "
                        f"retries. Please wait 1-2 minutes and try again. "
                        f"Consider upgrading from the free tier for higher limits."
                    ) from e
            else:
                raise


def _format_docs(docs):
    """Format retrieved documents into a single context string."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[Source: {source}, Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


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
    Build a conversational RAG chain using LCEL.

    The chain:
    1. Reformulates the user question using chat history for context
    2. Retrieves relevant document chunks from ChromaDB
    3. Generates a grounded answer using Gemini with source citations
    """
    llm = _get_llm()
    retriever = get_retriever()

    # Contextualize chain: reformulate question with history
    contextualize_chain = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

    def contextualized_question(input_dict):
        """If there's chat history, reformulate the question."""
        if input_dict.get("chat_history"):
            return contextualize_chain.invoke(input_dict)
        return input_dict["input"]

    # Full RAG chain using LCEL
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: retriever.invoke(contextualized_question(x))
        )
    )

    return rag_chain, llm


def query(question: str, chat_history: list) -> dict:
    """
    Query the RAG system with conversation history.

    Args:
        question: The user's question.
        chat_history: List of (human_msg, ai_msg) tuples.

    Returns:
        dict with 'answer' and 'context' (source documents).
    """
    rag_chain, llm = create_rag_chain()

    # Convert chat history to LangChain message format
    history_messages = []
    for human_msg, ai_msg in chat_history:
        history_messages.append(HumanMessage(content=human_msg))
        history_messages.append(AIMessage(content=ai_msg))

    # Step 1: Get context (retrieved docs)
    result = rag_chain.invoke(
        {"input": question, "chat_history": history_messages}
    )

    context_docs = result["context"]

    # Step 2: Generate answer with context
    formatted_context = _format_docs(context_docs)
    qa_messages = QA_PROMPT.invoke(
        {
            "context": formatted_context,
            "chat_history": history_messages,
            "input": question,
        }
    )

    answer = _invoke_with_retry(llm, qa_messages)

    return {
        "answer": answer.content,
        "context": context_docs,
    }
