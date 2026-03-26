"""
RAG-Powered Document Q&A System
Upload PDFs/DOCX files and ask questions using Retrieval-Augmented Generation.
"""

import streamlit as st
import os
from rag.document_loader import load_document
from rag.vector_store import add_documents, delete_collection, get_vector_store
from rag.query_engine import query

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI – RAG Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ───────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────
def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0


init_session_state()


# ── Sidebar ───────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with document upload and management."""
    with st.sidebar:
        # Header
        st.markdown("## 🧠 DocuMind AI")
        st.markdown(
            "<p style='color: #8b8b9e; font-size: 0.85rem; margin-top: -10px;'>"
            "RAG-Powered Document Intelligence</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # File Uploader
        st.markdown("### 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drop your files here",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="file_uploader",
            help="Supports PDF and DOCX files. Upload multiple files at once.",
        )

        # Process uploaded files
        if uploaded_files:
            new_files = [
                f
                for f in uploaded_files
                if f.name not in st.session_state.uploaded_docs
            ]

            if new_files:
                with st.spinner("🔄 Processing documents..."):
                    for file in new_files:
                        try:
                            docs = load_document(file)
                            if docs:
                                add_documents(docs)
                                st.session_state.uploaded_docs.append(file.name)
                                st.session_state.doc_count += 1
                                st.success(f"✅ {file.name}", icon="📄")
                            else:
                                st.warning(
                                    f"⚠️ No text found in {file.name}", icon="⚠️"
                                )
                        except Exception as e:
                            st.error(f"❌ Error processing {file.name}: {str(e)}")

        st.divider()

        # Uploaded Documents List
        if st.session_state.uploaded_docs:
            st.markdown("### 📚 Loaded Documents")
            for i, doc_name in enumerate(st.session_state.uploaded_docs):
                st.markdown(
                    f"<div style='background: rgba(255,255,255,0.04); "
                    f"border: 1px solid rgba(255,255,255,0.08); "
                    f"border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; "
                    f"font-size: 0.85rem; color: #e8e8ed;'>"
                    f"📄 {doc_name}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # Clear All button
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                try:
                    delete_collection()
                    st.session_state.uploaded_docs = []
                    st.session_state.chat_history = []
                    st.session_state.messages = []
                    st.session_state.doc_count = 0
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing: {str(e)}")
        else:
            st.markdown(
                "<div style='text-align: center; color: #5a5a6e; padding: 2rem 0;'>"
                "<p style='font-size: 2rem; margin-bottom: 0.5rem;'>📂</p>"
                "<p style='font-size: 0.85rem;'>No documents uploaded yet.<br>"
                "Upload a PDF or DOCX to get started.</p></div>",
                unsafe_allow_html=True,
            )

        # Footer
        st.divider()
        st.markdown(
            "<div style='text-align: center; color: #5a5a6e; font-size: 0.75rem;'>"
            "Powered by LangChain + Gemini + ChromaDB</div>",
            unsafe_allow_html=True,
        )


# ── Main Chat Area ────────────────────────────────────────────────────
def render_chat():
    """Render the main chat interface."""

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            "<h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
            "-webkit-background-clip: text; -webkit-text-fill-color: transparent; "
            "font-size: 2.2rem; font-weight: 700; margin-bottom: 0;'>"
            "DocuMind AI</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color: #8b8b9e; font-size: 1rem; margin-top: -5px;'>"
            "Ask anything about your uploaded documents</p>",
            unsafe_allow_html=True,
        )
    with col2:
        if st.session_state.uploaded_docs:
            st.metric("Documents", st.session_state.doc_count)

    st.divider()

    # Welcome message when no docs uploaded
    if not st.session_state.uploaded_docs:
        st.markdown(
            "<div style='text-align: center; padding: 4rem 2rem;'>"
            "<p style='font-size: 4rem; margin-bottom: 1rem;'>🧠</p>"
            "<h2 style='color: #e8e8ed; font-weight: 600;'>Welcome to DocuMind AI</h2>"
            "<p style='color: #8b8b9e; font-size: 1.1rem; max-width: 500px; margin: 0 auto;'>"
            "Upload a PDF or DOCX document in the sidebar to start asking questions. "
            "I'll use RAG to find relevant context and give you accurate, "
            "source-cited answers.</p>"
            "<div style='margin-top: 2rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;'>"
            "<span style='background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.2); "
            "border-radius: 20px; padding: 6px 16px; color: #667eea; font-size: 0.85rem;'>"
            "📄 Multi-format support</span>"
            "<span style='background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.2); "
            "border-radius: 20px; padding: 6px 16px; color: #34d399; font-size: 0.85rem;'>"
            "🔍 Context-aware answers</span>"
            "<span style='background: rgba(118,75,162,0.1); border: 1px solid rgba(118,75,162,0.2); "
            "border-radius: 20px; padding: 6px 16px; color: #a78bfa; font-size: 0.85rem;'>"
            "💬 Conversational memory</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 View Sources", expanded=False):
                    for j, source in enumerate(message["sources"]):
                        src_name = source.metadata.get("source", "Unknown")
                        src_page = source.metadata.get("page", "N/A")
                        st.markdown(
                            f"**Source {j+1}**: {src_name} (Page {src_page})"
                        )
                        st.caption(source.page_content[:300] + "...")
                        if j < len(message["sources"]) - 1:
                            st.divider()

    # Chat input
    if user_input := st.chat_input(
        "Ask a question about your documents...",
        key="chat_input",
    ):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating answer..."):
                try:
                    result = query(user_input, st.session_state.chat_history)
                    answer = result["answer"]
                    sources = result["context"]

                    st.markdown(answer)

                    # Show source citations
                    if sources:
                        with st.expander("📚 View Sources", expanded=False):
                            for j, source in enumerate(sources):
                                src_name = source.metadata.get("source", "Unknown")
                                src_page = source.metadata.get("page", "N/A")
                                st.markdown(
                                    f"**Source {j+1}**: {src_name} (Page {src_page})"
                                )
                                st.caption(source.page_content[:300] + "...")
                                if j < len(sources) - 1:
                                    st.divider()

                    # Update history
                    st.session_state.chat_history.append((user_input, answer))
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as e:
                    error_msg = f"❌ Error generating answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "sources": []}
                    )


# ── Main ──────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
else:
    main()
