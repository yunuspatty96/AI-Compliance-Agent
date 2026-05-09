# ============================================================
# STREAMLIT WEB APP — Data Privacy Compliance Agent (Local Version)
# No API Key Required
# ============================================================

import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
import os
import gdown
import torch

# ── Page configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Privacy Compliance Agent (Local)",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0 0; font-size: 1rem; }
    .result-card {
        background: #0f1923; border: 1px solid #1e3a5f; border-radius: 10px; padding: 1.5rem;
        font-family: monospace; white-space: pre-wrap; color: #e0e0e0; line-height: 1.7;
    }
    .sidebar-section { background: #1a1a2e; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; border-left: 3px solid #e94560; }
</style>
""", unsafe_allow_html=True)

# ── Document sources ─────────────────────────────────────────
DOCUMENTS = {
    "UU_PDP.pdf":             "1gwzDwZZoqorirb9XXRE-acXHY9URYIrB",
    "UU_ITE.pdf":             "1DSdMaL2cJGO4__m2CbB1MKDRDnsBb56M",
    "UU_ITE_AmandemenI.pdf":  "1P9C3y6TG98CK9ObB_Ji-nIwP9c_2YfYw",
    "UU_ITE_AmandemenII.pdf": "1TLL4pw8Bk1Q3BOTn7EHOqtRd2X4nsRW4",
    "ALU_Regulations.pdf":    "1pcHYYEUYSdwYPS1Y9DrdWEbqqm0yJ1uk",
}

DOC_LABELS = {
    "UU_PDP.pdf":             "Personal Data Protection Law (UU PDP)",
    "UU_ITE.pdf":             "Electronic Information & Transactions Law (UU ITE)",
    "UU_ITE_AmandemenI.pdf":  "UU ITE Amendment I",
    "UU_ITE_AmandemenII.pdf": "UU ITE Amendment II",
    "ALU_Regulations.pdf":    "ALU University Regulations",
}

# ── Index & Model Helpers ────────────────────────────────────
def index_exists() -> bool:
    return os.path.exists("faiss_index/index.faiss")

def download_documents(status_placeholder) -> list:
    os.makedirs("documents", exist_ok=True)
    downloaded = []
    for filename, file_id in DOCUMENTS.items():
        path = f"documents/{filename}"
        if not os.path.exists(path):
            status_placeholder.info(f"⬇️ Downloading {DOC_LABELS[filename]}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=True)
        downloaded.append(path)
    return downloaded

def build_index(status_placeholder):
    status_placeholder.info("📖 Loading PDF documents...")
    all_docs = []
    for filename, label in DOC_LABELS.items():
        path = f"documents/{filename}"
        if not os.path.exists(path): continue
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["source_name"] = label
        all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    
    status_placeholder.info("🔢 Building vector index (Local Embeddings)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs("faiss_index", exist_ok=True)
    vectorstore.save_local("faiss_index")
    return True

# ── Auto-setup on first run ──────────────────────────────────
if not index_exists():
    st.info("⚙️ **First-time setup:** Downloading documents and building knowledge base...")
    status = st.empty()
    download_documents(status)
    if build_index(status):
        st.success("✅ Knowledge base ready!")
        st.rerun()
    st.stop()

# ── Session State ─────────────────────────────────────────────
if "messages" not in st.session_state: st.session_state.messages = []
if "last_report" not in st.session_state: st.session_state.last_report = None

# ── Local Model Loader ────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Local AI Model (this may take a minute)...")
def load_system():
    # Using FLAN-T5-Base: Free, No Key, Good for RAG, fits in memory
    model_id = "google/flan-t5-base" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.1,
        device=-1 # Set to 0 if GPU is available
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def retrieve_docs(question: str) -> str:
        docs = retriever.invoke(question)
        return "\n".join([f"Source: {d.metadata['source_name']}\n{d.page_content}" for d in docs])

    # Simple Prompt for Local Models
    PROMPT_TEMPLATE = """Answer the following question based ONLY on the provided legal context. 
If the answer is not in the context, say "No relevant clauses found."

Context:
{context}

Question: {question}
Analysis:"""

    REPORT_PROMPT = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    report_chain = (
        {"context": lambda x: retrieve_docs(x["question"]), "question": lambda x: x["question"]}
        | REPORT_PROMPT | llm | StrOutputParser()
    )
    
    return report_chain

# ── UI Layout ────────────────────────────────────────────────
st.markdown("""<div class="main-header"><h1>🔐 AI COMPLIANCE AGENT (LOCAL)</h1><p>Privacy Risk Analyst running locally without API keys</p></div>""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown("### 📥 Input Scenario")
    scenario = st.text_area("Describe the privacy scenario:", height=160)
    analyze_btn = st.button("🔍 Analyze Scenario", type="primary", use_container_width=True)

with col_right:
    tab_report, tab_chat = st.tabs(["📊 Analysis Report", "💬 Info"])
    with tab_report:
        if analyze_btn and scenario:
            with st.spinner("Analyzing locally..."):
                agent = load_system()
                st.session_state.last_report = agent.invoke({"question": scenario})
        
        if st.session_state.last_report:
            st.markdown(f'<div class="result-card">{st.session_state.last_report}</div>', unsafe_allow_html=True)
        else:
            st.info("Submit a scenario to generate a report.")
    
    with tab_chat:
        st.write("This agent is running a local model (**Flan-T5**). It does not send data to external APIs like Groq or OpenAI.")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Status")
    st.success("✅ Running Offline Mode")
    st.info("💡 Note: Local inference is slower than API-based models but more private.")
    if st.button("🗑️ Clear Data"):
        st.session_state.messages = []
        st.rerun()

st.divider()
st.markdown('<div style="text-align:center; color:#808080; font-size:0.8rem;">Developed by Yunus P. | Powered by Local Transformers</div>', unsafe_allow_html=True)