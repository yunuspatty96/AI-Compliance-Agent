
# ============================================================
# STREAMLIT WEB APP — Data Privacy Compliance Agent
# Deploy with: streamlit run app.py
# ============================================================

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
import os
import gdown

# ── Load API key from environment ────────────────────────────
api_key = os.environ.get("GROQ_API_KEY", "")

# ── Page configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Privacy Compliance Agent",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for a polished look ──────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0 0; font-size: 1rem; }

    .risk-critical { background:#ff4444; color:white; padding:6px 16px; border-radius:20px; font-weight:bold; }
    .risk-high     { background:#ff8800; color:white; padding:6px 16px; border-radius:20px; font-weight:bold; }
    .risk-medium   { background:#ffbb00; color:black; padding:6px 16px; border-radius:20px; font-weight:bold; }
    .risk-low      { background:#00bb44; color:white; padding:6px 16px; border-radius:20px; font-weight:bold; }

    .result-card {
        background: #0f1923;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1.5rem;
        font-family: monospace;
        white-space: pre-wrap;
        color: #e0e0e0;
        line-height: 1.7;
    }

    .sidebar-section {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 3px solid #e94560;
    }

    .metric-box {
        text-align: center;
        background: #16213e;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #0f3460;
    }
    .metric-box .number { font-size: 2rem; font-weight: bold; color: #e94560; }
    .metric-box .label  { font-size: 0.85rem; color: #a8b2d8; }
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

# ── Index helpers ────────────────────────────────────────────
def index_exists() -> bool:
    return os.path.exists("faiss_index/index.faiss") and os.path.exists("faiss_index/index.pkl")

def download_documents(status_placeholder) -> list:
    os.makedirs("documents", exist_ok=True)
    downloaded = []
    for filename, file_id in DOCUMENTS.items():
        path = f"documents/{filename}"
        if os.path.exists(path):
            downloaded.append(path)
            continue
        try:
            status_placeholder.info(f"⬇️ Downloading {DOC_LABELS[filename]}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=True)
            if os.path.exists(path):
                downloaded.append(path)
            else:
                status_placeholder.warning(f"⚠️ Could not download {filename}")
        except Exception as e:
            status_placeholder.warning(f"⚠️ Error downloading {filename}: {e}")
    return downloaded

def build_index(status_placeholder):
    status_placeholder.info("📖 Loading PDF documents...")
    all_docs = []
    for filename, label in DOC_LABELS.items():
        path = f"documents/{filename}"
        if not os.path.exists(path):
            continue
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            for page in pages:
                page.metadata["source_name"] = label
                page.metadata["source_file"] = filename
            all_docs.extend(pages)
        except Exception as e:
            status_placeholder.warning(f"⚠️ Could not read {filename}: {e}")

    if not all_docs:
        status_placeholder.error("❌ No documents could be loaded. Cannot build index.")
        return False

    status_placeholder.info(f"✂️ Splitting {len(all_docs)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "Article ", "Section ", ". ", " "]
    )
    chunks = splitter.split_documents(all_docs)

    status_placeholder.info(f"🔢 Building vector index from {len(chunks)} chunks (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs("faiss_index", exist_ok=True)
    vectorstore.save_local("faiss_index")
    return True

# ── Auto-setup on first run ──────────────────────────────────
if not index_exists():
    st.markdown("""
    <div class="main-header">
        <h1>🔐 AI COMPLIANCE AGENT</h1>
        <p>A RAG-based data protection and privacy risk analyst agent for university</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("⚙️ **First-time setup:** Downloading legal documents and building the knowledge base. This only happens once and takes about 1–2 minutes.")
    status = st.empty()
    progress = st.progress(0)

    status.info("⬇️ Downloading legal documents from source...")
    progress.progress(10)
    download_documents(status)
    progress.progress(40)

    success = build_index(status)
    progress.progress(90)

    if success:
        progress.progress(100)
        status.success("✅ Knowledge base ready! Reloading the app...")
        st.rerun()
    else:
        status.error("❌ Setup failed. Please check your internet connection and try reloading.")
    st.stop()

# ── Session State Initialization ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None

# ── App header ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔐 AI COMPLIANCE AGENT</h1>
    <p>A RAG-based data protection and privacy risk analyst agent for university</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: settings and info ──────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    if api_key:
        st.success("✅ AI model connected and ready")
    else:
        st.error("❌ GROQ_API_KEY not configured")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### 📚 Knowledge Base")
    st.markdown("""
    <div class="sidebar-section" style="color: white;">
    This agent analyzes scenarios against:
    <br><br>
    📜 <b>UU PDP</b> — Personal Data Protection Law<br>
    💻 <b>UU ITE</b> — Electronic Information Law<br>
    📝 <b>UU ITE Amendment I</b><br>
    📝 <b>UU ITE Amendment II</b><br>
    🏫 <b>ALU University Regulations</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    doc_count = len([f for f in DOCUMENTS if os.path.exists(f"documents/{f}")])
    st.markdown(f"**📂 Documents loaded:** {doc_count} / {len(DOCUMENTS)}")
    st.markdown("**🗄️ Index status:** ✅ Ready" if index_exists() else "**🗄️ Index status:** ⚠️ Missing")

# ── Main layout: two columns ────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown("### 📥 Input Scenario")
    examples = {
        "📱 Phone Number Sale": "A lecturer sold student phone numbers to a third-party without consent.",
        "📊 Grade Data Leak": "IT staff accidentally published student grades and ID numbers on the public website.",
        "📷 Unauthorized CCTV": "University installed CCTV in dorm rooms without notifying students."
    }

    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["scenario_input"] = text

    scenario = st.text_area("Describe the privacy scenario:", value=st.session_state.get("scenario_input", ""), height=160, key="scenario_input")
    analyze_btn = st.button("🔍 Analyze Scenario", type="primary", use_container_width=True)

# ── Load models and vector store ────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system(api_key: str):
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 7})

    def retrieve_docs(question: str) -> str:
        docs = retriever.invoke(question)
        context = ""
        for doc in docs:
            context += f"\nSource: {doc.metadata.get('source_name')} (Page {doc.metadata.get('page')})\n{doc.page_content}\n"
        return context

    REPORT_PROMPT = ChatPromptTemplate.from_template("""
You are an AI Agent acting as a Data Privacy & Compliance Analyst for a university in Indonesia.

Your responsibilities:
1. Analyze privacy and data protection incidents carefully
2. Assess institutional and legal risks objectively
3. Identify ALL relevant articles and clauses from the provided legal documents
4. Cross-reference findings across MULTIPLE documents when applicable
5. Provide concrete, actionable recommendations

IMPORTANT RULES:
- Use ONLY the provided legal/document context
- Do NOT invent legal references
- If no relevant clause exists, explicitly state: "No directly applicable clauses found."
- Maintain formal, professional English
- Focus specifically on university/government/educational data privacy compliance

Provide your analysis in this exact format:

╔════════════════════════════════════════════╗

🔍 DATA PROTECTION & PRIVACY RISK ANALYSIS REPORT

📌 SITUATION SUMMARY:
[Summarize the situation in 2–3 sentences, identifying the key privacy issue]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ RISK LEVEL: [LOW / MEDIUM / HIGH / CRITICAL]

Justification:
- Severity Score: [1–5]
- Likelihood Score: [1–5]
- Risk Matrix Result: [Final Level]

Explain clearly why this level was assigned based on severity and likelihood.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 APPLICABLE LAWS & REGULATIONS:

From Personal Data Protection Law (UU PDP):
• [List relevant articles/clauses and explain relevance]

From Electronic Information Law (UU ITE & Amendments):
• [List relevant articles/clauses and explain relevance]

From ALU University Regulations:
• [List relevant sections and explain relevance]

If a source contains no relevant clauses, explicitly state: "No directly applicable clauses found."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 VIOLATIONS IDENTIFIED:

[List each violation clearly]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚖️ APPLICABLE SANCTIONS:

[List applicable sanctions from each relevant legal source]

Include:
- Criminal penalties
- Administrative sanctions
- Financial penalties/fines
- Institutional disciplinary actions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ RECOMMENDED ACTIONS:

Provide BOTH:
- Immediate corrective actions
- Long-term preventive measures

Use numbered actionable recommendations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENT SOURCES USED IN THIS ANALYSIS:

[List all contributing documents and page references if available]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 COMPARISON WITH PREVIOUS CASES (if applicable):

[Compare with previous incidents from chat history if relevant]

╚══════════════════════════════════════════════════════════╝

Legal context:
{context}

Scenario to analyze:
{question}
    """)

    CHAT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an AI Agent acting as a Data Privacy & Compliance Analyst for a university in Indonesia.

Your responsibilities:
1. Analyze privacy and data protection incidents carefully
2. Assess institutional and legal risks objectively
3. Identify ALL relevant articles and clauses from the provided legal documents
4. Cross-reference findings across MULTIPLE documents when applicable
5. Provide concrete, actionable recommendations
6. Maintain conversation continuity using previous chat history when relevant

IMPORTANT RULES:
- Use ONLY the provided legal/document context
- Do NOT invent legal references
- If no relevant clause exists, explicitly state: "No directly applicable clauses found."
- Maintain formal, professional English
- Always explain reasoning clearly
- Focus specifically on university/government/educational data privacy compliance
- DO NOT use the structured report template
- Explain things interactively and be concise and practical
- Still reference relevant laws if needed
- Ask follow-up questions when appropriate

IMPORTANT:
- Base the analysis ONLY on provided context
- Do NOT use external legal assumptions
- Cite article numbers precisely
- Cross-reference multiple documents whenever possible
- If evidence is insufficient, clearly state limitations"""),

        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Legal context:\n{context}\n\nQuestion: {question}")
    ])

    report_chain = (
        {
            "context": lambda x: retrieve_docs(x["question"]),
            "question": lambda x: x["question"]
        }
        | REPORT_PROMPT | llm | StrOutputParser()
    )
    chat_chain = (
        {
            "context": lambda x: retrieve_docs(x["question"]),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | CHAT_PROMPT | llm | StrOutputParser()
    )

    return report_chain, chat_chain

# ── Tabs for Analysis vs Chat ───────────────────────────────
with col_right:
    tab_report, tab_chat = st.tabs(["📊 Analysis Report", "💬 Discussion Chat"])

    with tab_report:
        if analyze_btn and scenario:
            if not api_key:
                st.error("AI model not configured. Please contact the administrator.")
            else:
                with st.spinner("Analyzing scenario against legal documents..."):
                    report_agent, _ = load_system(api_key)
                    st.session_state.last_report = report_agent.invoke({"question": scenario})

        if st.session_state.last_report:
            st.markdown(f'<div class="result-card">{st.session_state.last_report}</div>', unsafe_allow_html=True)
        else:
            st.info("Submit a scenario to generate a report.")

    with tab_chat:
        st.caption("Got questions? Ask away! Let's chat about the details.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me anything about data protection & privacy regulations..."):
            if not api_key:
                st.error("AI model not configured. Please contact the administrator.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    _, chat_agent = load_system(api_key)
                    history = []
                    for m in st.session_state.messages[:-1]:
                        if m["role"] == "user":
                            history.append(HumanMessage(content=m["content"]))
                        else:
                            history.append(AIMessage(content=m["content"]))

                    response = chat_agent.invoke({"question": prompt, "chat_history": history})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown('<div style="text-align:center; color:#808080; font-size:0.8rem;">Developed by Yunus P. | Powered by LLaMA 3.3 | For educational and compliance guidance purposes only. Not a substitute for qualified legal counsel</div>', unsafe_allow_html=True)
