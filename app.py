import os
import tempfile
import logging
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flashrank import Ranker, RerankRequest

DB_PATH = "./data/chroma_db_vectors"
COLLECTION_NAME = "hardware_manuals"

# -----------------------------------------------------------------------------
# 1. SETUP & AUTOMATIC CREDENTIAL DETECTION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="RAG APP", page_icon="⚙️", layout="wide")
st.title("Enterprise Hardware Diagnostics RAG Assistant ⚙️")

# Resolve Groq API Key automatically from Streamlit Secrets or .env
default_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_reranker():
    return Ranker(model_name="ms-marco-TinyBERT-L-2-v2")

def load_existing_vector_store():
    """Auto-loads ChromaDB from disk if previous embeddings exist."""
    if os.path.exists(DB_PATH):
        embeddings = get_embedding_model()
        store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        # Check if the collection contains indexed records
        if store._collection.count() > 0:
            return store
    return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_existing_vector_store()

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONFIGURATION & OPTIONAL INGESTION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔧 Configuration")
    
    if default_api_key:
        st.success("✅ Groq API Key detected from environment.")
        groq_api_key = default_api_key
    else:
        groq_api_key = st.text_input("Groq API Key", type="password")

    selected_model = st.selectbox(
        "Inference Model", 
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    )

    st.divider()
    st.header("📂 Knowledge Base Status")
    
    if st.session_state.vector_store:
        doc_count = st.session_state.vector_store._collection.count()
        st.info(f"🟢 Active Index: **{doc_count} chunks** in memory.")
    else:
        st.warning("⚪ No existing vector index found on disk.")

    uploaded_files = st.file_uploader(
        "Add / Update PDF Manuals", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    process_btn = st.button("Process & Index", type="primary")

# -----------------------------------------------------------------------------
# 3. DOCUMENT INGESTION LOGIC
# -----------------------------------------------------------------------------
def process_and_index_pdfs(files):
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = file.name

        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)
        os.remove(tmp_path)

    embeddings = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME
    )
    return vector_store

def retrieve_and_rerank(query, vector_store, top_k=8, top_n=3):
    initial_docs = vector_store.similarity_search(query, k=top_k)
    if not initial_docs:
        return []

    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(initial_docs)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    ranker = get_reranker()
    ranked_results = ranker.rerank(rerank_request)
    return ranked_results[:top_n]

if process_btn and uploaded_files:
    with st.spinner("Indexing new documents into persistent store..."):
        st.session_state.vector_store = process_and_index_pdfs(uploaded_files)
        st.rerun()

# -----------------------------------------------------------------------------
# 4. CHAT INTERFACE & GENERATION
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 View Verified Source Citations"):
                for src in msg["sources"]:
                    st.caption(f"**Source:** {src['source']} | **Page:** {src['page']} | **Re-rank Score:** {src['score']:.4f}")
                    st.text(src["text"])

if query := st.chat_input("Ask a hardware spec or diagnostic question..."):
    if not groq_api_key:
        st.error("Please provide a valid Groq API Key.")
        st.stop()
    if not st.session_state.vector_store:
        st.warning("Please upload and index a document first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    llm = ChatGroq(model=selected_model, groq_api_key=groq_api_key, temperature=0.0)

    system_prompt = (
        "You are an expert PC hardware diagnostic engineer. Answer the user's question "
        "using strictly the technical manual excerpts provided in the context below. "
        "If the answer cannot be directly determined from the context, state explicitly: "
        "'I cannot find this information in the provided documentation.'\n\n"
        "Technical Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and re-ranking relevant manual excerpts..."):
            ranked_results = retrieve_and_rerank(query, st.session_state.vector_store)

            context_blocks = []
            source_data = []
            for i, result in enumerate(ranked_results, start=1):
                meta = result.get("meta", {})
                doc_name = meta.get("source", "Document.pdf")
                page_num = meta.get("page", 0) + 1
                score = result.get("score", 0.0)
                text = result.get("text", "")

                context_blocks.append(f"--- [Citation {i}: {doc_name} (Page {page_num})] ---\n{text}")
                source_data.append({"source": doc_name, "page": page_num, "score": score, "text": text})

            formatted_context = "\n\n".join(context_blocks)

            rag_chain = prompt | llm | StrOutputParser()
            answer = rag_chain.invoke({
                "context": formatted_context,
                "question": query
            })

            st.markdown(answer)

            with st.expander("🔍 View Verified Source Citations"):
                for src in source_data:
                    st.caption(f"**Source:** {src['source']} | **Page:** {src['page']} | **Re-rank Score:** {src['score']:.4f}")
                    st.text(src["text"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_data
            })