# ============================================================
# STREAMLIT WEB APP — Data Privacy Compliance Agent
# NO API KEY REQUIRED VERSION
# Run locally:
#   pip install streamlit langchain langchain-community
#   pip install sentence-transformers faiss-cpu transformers torch
#   pip install pypdf gdown
#
# Start:
#   streamlit run app_no_api.py
# ============================================================

import streamlit as st
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
import os
import gdown

# ── Page configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Privacy Compliance Agent",
    page_icon="🔐",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 2rem;
}
.main-header h1 { color: #e94560; margin: 0; }
.main-header p  { color: #a8b2d8; }

.result-card {
    background: #0f1923;
    border-radius: 10px;
    padding: 1rem;
    color: white;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ── Google Drive Documents ──────────────────────────────────
DOCUMENTS = {
    "UU_PDP.pdf":             "1gwzDwZZoqorirb9XXRE-acXHY9URYIrB",
    "UU_ITE.pdf":             "1DSdMaL2cJGO4__m2CbB1MKDRDnsBb56M",
    "UU_ITE_AmandemenI.pdf":  "1P9C3y6TG98CK9ObB_Ji-nIwP9c_2YfYw",
    "UU_ITE_AmandemenII.pdf": "1TLL4pw8Bk1Q3BOTn7EHOqtRd2X4nsRW4",
    "ALU_Regulations.pdf":    "1pcHYYEUYSdwYPS1Y9DrdWEbqqm0yJ1uk",
}

# ── Helper Functions ────────────────────────────────────────
def index_exists():
    return os.path.exists("faiss_index/index.faiss")

def download_documents():
    os.makedirs("documents", exist_ok=True)

    for filename, file_id in DOCUMENTS.items():
        path = f"documents/{filename}"

        if not os.path.exists(path):
            with st.spinner(f"Downloading {filename}..."):
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    path,
                    quiet=True
                )

def build_index():
    all_docs = []

    for filename in DOCUMENTS.keys():
        path = f"documents/{filename}"

        if os.path.exists(path):
            loader = PyPDFLoader(path)
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = filename

            all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs("faiss_index", exist_ok=True)
    vectorstore.save_local("faiss_index")

# ── First Run Setup ─────────────────────────────────────────
if not index_exists():
    st.info("First setup: downloading documents and building index...")
    download_documents()
    build_index()
    st.success("Knowledge base ready!")
    st.rerun()

# ── Load Local AI Model (NO API KEY) ───────────────────────
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=512,
        temperature=0.2
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    return llm, retriever

llm, retriever = load_llm()

# ── Prompt ──────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_template("""
You are a Data Privacy & Compliance Analyst for a university in Indonesia.

Use ONLY the provided legal context.

LEGAL CONTEXT:
{context}

SCENARIO:
{question}

Provide:
1. Situation Summary
2. Risk Level
3. Relevant Laws
4. Violations
5. Recommended Actions

Answer professionally and clearly.
""")

chain = PROMPT | llm | StrOutputParser()

# ── Session State ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
<h1>🔐 AI COMPLIANCE AGENT</h1>
<p>No API Key Required Version</p>
</div>
""", unsafe_allow_html=True)

# ── Main Layout ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Analysis", "💬 Chat"])

with tab1:
    st.subheader("Privacy Scenario Analysis")

    scenario = st.text_area(
        "Describe the incident:",
        height=180,
        placeholder="Example: Lecturer leaked student phone numbers..."
    )

    if st.button("Analyze Scenario", type="primary"):

        if scenario.strip():

            with st.spinner("Analyzing..."):

                docs = retriever.invoke(scenario)

                context = ""

                for doc in docs:
                    context += f"""
Source: {doc.metadata.get('source')}
Content:
{doc.page_content}

"""

                response = chain.invoke({
                    "context": context,
                    "question": scenario
                })

                st.markdown(
                    f'<div class="result-card">{response}</div>',
                    unsafe_allow_html=True
                )

with tab2:

    st.subheader("Discussion Chat")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question...")

    if user_input:

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            docs = retriever.invoke(user_input)

            context = ""

            for doc in docs:
                context += doc.page_content + "\n\n"

            response = chain.invoke({
                "context": context,
                "question": user_input
            })

            st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

# ── Footer ──────────────────────────────────────────────────
st.divider()

st.caption(
    "Developed for educational and compliance guidance purposes."
)
