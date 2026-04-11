"""
DocuMind AI – RAG-Powered Document Q&A System
Upload PDFs/DOCX files and ask questions using Retrieval-Augmented Generation.
"""

import streamlit as st
import os
import base64
from rag.document_loader import load_document
from rag.vector_store import add_documents, delete_collection, get_vector_store
from rag.query_engine import query

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI – RAG Document Q&A",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ───────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _get_hero_image_base64():
    """Load hero background image as base64 for inline embedding."""
    hero_path = os.path.join(os.path.dirname(__file__), "assets", "hero_bg.png")
    if os.path.exists(hero_path):
        with open(hero_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


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
        # Brand Header
        st.markdown(
            '<div class="sidebar-brand">'
            '  <div class="sidebar-brand-icon">🌿</div>'
            "  <div>"
            '    <div class="sidebar-brand-text">DocuMind AI</div>'
            '    <div class="sidebar-brand-sub">Document Intelligence</div>'
            "  </div>"
            "</div>",
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
            help="Supports PDF and DOCX files (up to 200MB each).",
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
                                st.success(f"✅ {file.name}")
                            else:
                                st.warning(f"No text found in {file.name}")
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")

        st.divider()

        # Uploaded Documents List
        if st.session_state.uploaded_docs:
            st.markdown("### 📚 Loaded Documents")
            for doc_name in st.session_state.uploaded_docs:
                st.markdown(
                    f'<div class="doc-card">📄 {doc_name}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

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
                '<div style="text-align: center; color: #8a8a9a; padding: 2rem 0;">'
                '<p style="font-size: 2.2rem; margin-bottom: 0.5rem;">📂</p>'
                '<p style="font-size: 0.85rem; line-height: 1.6;">'
                "No documents uploaded yet.<br>"
                "Upload a PDF or DOCX to begin.</p></div>",
                unsafe_allow_html=True,
            )

        # Footer
        st.divider()
        st.markdown(
            '<div class="footer-text">'
            "Built with LangChain · Gemini · ChromaDB</div>",
            unsafe_allow_html=True,
        )


# ── Home Screen (Hero) ───────────────────────────────────────────────
def render_home():
    """Render the hero home screen when no documents are uploaded."""
    hero_b64 = _get_hero_image_base64()

    if hero_b64:
        st.markdown(
            f"""
            <div class="hero-container">
                <img src="data:image/png;base64,{hero_b64}" class="hero-image" alt="DocuMind AI" />
                <div class="hero-overlay">
                    <div class="hero-title">DocuMind AI</div>
                    <div class="hero-subtitle">
                        Upload your research papers and documents, then ask questions
                        — powered by RAG for accurate, source-cited answers.
                    </div>
                    <div class="feature-pills">
                        <span class="pill">📄 PDF & DOCX</span>
                        <span class="pill">🔍 Semantic Search</span>
                        <span class="pill">💬 Conversational</span>
                        <span class="pill">📚 Source Citations</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Fallback if hero image not found
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #6bb5a0 0%, #3d8b7a 40%, #2d6b5e 100%);
                        border-radius: 24px; padding: 4rem 2rem; text-align: center; margin-bottom: 2rem;">
                <div class="hero-title">DocuMind AI</div>
                <div class="hero-subtitle">
                    Upload your research papers and documents, then ask questions
                    — powered by RAG for accurate, source-cited answers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # How It Works cards
    st.markdown(
        """
        <div class="steps-grid">
            <div class="step-card">
                <span class="step-icon">📄</span>
                <h4>Upload Documents</h4>
                <p>Drop PDF or DOCX files in the sidebar. They're chunked and embedded automatically.</p>
            </div>
            <div class="step-card">
                <span class="step-icon" style="animation-delay: 0.5s;">🔍</span>
                <h4>Ask Questions</h4>
                <p>Type natural language queries. Our RAG engine finds the most relevant passages.</p>
            </div>
            <div class="step-card">
                <span class="step-icon" style="animation-delay: 1s;">✨</span>
                <h4>Get Cited Answers</h4>
                <p>Receive grounded answers with exact page and document source citations.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Chat Interface ────────────────────────────────────────────────────
def render_chat():
    """Render the conversational chat interface after documents are uploaded."""

    # Chat Header Bar
    st.markdown(
        f"""
        <div class="chat-header">
            <div>
                <div class="chat-header-title">🌿 DocuMind AI</div>
                <div class="chat-header-sub">Ask anything about your {st.session_state.doc_count} loaded document{'s' if st.session_state.doc_count != 1 else ''}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    if st.session_state.uploaded_docs:
        render_chat()
    else:
        render_home()


if __name__ == "__main__":
    main()
else:
    main()
