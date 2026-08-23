import os
import glob
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flashrank import Ranker, RerankRequest

DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "n8n_knowledge"
KNOWLEDGE_DIR = "./data"

# =============================================================================
# SAFE CREDENTIAL RESOLUTION & CACHED RESOURCES
# =============================================================================
def get_groq_api_key():
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return ""

@st.cache_resource(show_spinner=False)
def get_embedding_engine():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_reranker():
    return Ranker(model_name="ms-marco-TinyBERT-L-2-v2")

# =============================================================================
# VECTOR DATABASE PERSISTENCE & AUTO-INGESTION
# =============================================================================
def build_or_load_vectorstore():
    """Loads existing Chroma store or auto-indexes knowledge_base/*.txt files on startup."""
    embeddings = get_embedding_engine()

    # Load from disk if index exists
    if os.path.exists(DB_PATH):
        try:
            store = Chroma(
                persist_directory=DB_PATH,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )
            if store._collection.count() > 0:
                return store
        except Exception:
            pass

    # Ingest from data folder on first startup
    raw_docs = []
    if os.path.exists(KNOWLEDGE_DIR):
        for filepath in glob.glob(f"{KNOWLEDGE_DIR}/*.txt"):
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    raw_docs.append({"content": text, "source": filename})

    if not raw_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=420,
        chunk_overlap=50,
        separators=["\n\n", "\n", "- ", ". ", " "]
    )

    texts = []
    metadatas = []
    for doc in raw_docs:
        chunks = splitter.split_text(doc["content"])
        for idx, chunk in enumerate(chunks, start=1):
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk_id": idx})

    store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME
    )
    return store

def add_uploaded_files_to_store(files, current_store):
    splitter = RecursiveCharacterTextSplitter(chunk_size=420, chunk_overlap=50)
    new_texts = []
    new_metadatas = []

    for file in files:
        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
            splits = splitter.split_text(text)
            for idx, s in enumerate(splits, start=1):
                new_texts.append(s)
                new_metadatas.append({"source": file.name, "chunk_id": idx})
        elif file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page_idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    splits = splitter.split_text(text)
                    for idx, s in enumerate(splits, start=1):
                        new_texts.append(s)
                        new_metadatas.append({"source": f"{file.name} (p.{page_idx})", "chunk_id": idx})

    if new_texts:
        if current_store is None:
            embeddings = get_embedding_engine()
            current_store = Chroma.from_texts(
                texts=new_texts,
                embedding=embeddings,
                metadatas=new_metadatas,
                persist_directory=DB_PATH,
                collection_name=COLLECTION_NAME
            )
        else:
            current_store.add_texts(texts=new_texts, metadatas=new_metadatas)
            
    return current_store

# =============================================================================
# TWO-STAGE RETRIEVAL & RE-RANKING 
# =============================================================================
def retrieve_and_rerank(query, store, top_k=8, top_n=3):
    initial_docs = store.similarity_search(query, k=top_k)
    if not initial_docs:
        return []

    passages = [
        {"id": idx, "text": doc.page_content, "meta": doc.metadata}
        for idx, doc in enumerate(initial_docs)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    ranker = get_reranker()
    ranked_results = ranker.rerank(rerank_request)
    return ranked_results[:top_n]

# =============================================================================
# STREAMLIT FRONT-END 
# =============================================================================
st.set_page_config(page_title="n8n Diagnostic Assistant", page_icon="⚡", layout="wide")
st.title("⚡ n8n Architecture & Automation Assistant")

groq_api_key = get_groq_api_key()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    with st.spinner("Loading persistent n8n knowledge base..."):
        st.session_state.vector_store = build_or_load_vectorstore()

# Sidebar Setup
with st.sidebar:
    st.header("🔧 Configuration")
    if groq_api_key:
        st.success("✅ Groq API Key detected")
    else:
        groq_api_key = st.text_input("Groq API Key", type="password")

    selected_model = st.selectbox(
        "Inference Model",
        ["groq/compound", "openai/gpt-oss-120b"]
    )

    st.divider()
    st.header("📂 Knowledge Base Status")
    if st.session_state.vector_store:
        count = st.session_state.vector_store._collection.count()
        st.info(f"🟢 Active Index: **{count} chunks** loaded.")
    else:
        st.warning("⚪ No documents found in `knowledge_base/`.")

    uploaded_files = st.file_uploader(
        "Append Documents (.txt / .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )
    if st.button("Index Uploaded Files", type="primary"):
        if uploaded_files:
            with st.spinner("Indexing new files..."):
                st.session_state.vector_store = add_uploaded_files_to_store(uploaded_files, st.session_state.vector_store)
                st.success("Documents appended to persistent store!")
                st.rerun()

# Render Conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 View Verified Source Citations"):
                for src in msg["sources"]:
                    st.caption(f"**Source:** `{src['source']}` | **Section:** `{src['chunk_id']}` | **Score:** `{src['score']:.4f}`")
                    st.text(src["text"])

# Handle Chat Input
if query := st.chat_input("Ask an n8n architecture, expression, or execution question..."):
    if not groq_api_key:
        st.error("Please provide a valid Groq API Key.")
        st.stop()
    if not st.session_state.vector_store:
        st.error("Vector store is uninitialized. Place `.txt` files in `knowledge_base/` and reload.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving facts & re-ranking relevance..."):
            ranked_results = retrieve_and_rerank(query, st.session_state.vector_store)

            context_blocks = []
            source_data = []
            for i, res in enumerate(ranked_results, start=1):
                meta = res.get("meta", {})
                doc_name = meta.get("source", "n8n_docs.txt")
                chunk_id = meta.get("chunk_id", 1)
                score = res.get("score", 0.0)
                text = res.get("text", "")

                context_blocks.append(f"--- [Citation {i}: {doc_name} (Section {chunk_id})] ---\n{text}")
                source_data.append({"source": doc_name, "chunk_id": chunk_id, "score": score, "text": text})

            formatted_context = "\n\n".join(context_blocks)

            # Grounding Guardrail Prompt 
            system_prompt = (
                "You are an expert n8n workflow automation engineer. Answer the user's question "
                "using strictly the verified documentation excerpts provided in the context below. "
                "If the answer cannot be directly determined from the context, state explicitly: "
                "'I cannot find this information in the provided documentation.' "
                "Never invent non-existent node parameters, syntax, or execution rules.\n\n"
                "Technical Context:\n{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])

            llm = ChatGroq(model=selected_model, groq_api_key=groq_api_key, temperature=0.0)
            rag_chain = prompt | llm | StrOutputParser()

            answer = rag_chain.invoke({
                "context": formatted_context,
                "question": query
            })

            st.markdown(answer)

            with st.expander("🔍 View Verified Source Citations"):
                for src in source_data:
                    st.caption(f"**Source:** `{src['source']}` | **Section:** `{src['chunk_id']}` | **Score:** `{src['score']:.4f}`")
                    st.text(src["text"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_data
            })