# ============================================================
# AI COMPLIANCE AGENT — Streamlit App
# Deploy with: streamlit run app.py
#
# Prerequisites:
#   pip install langchain langchain-core langchain-community \
#               langchain-text-splitters langchain-groq      \
#               faiss-cpu pypdf sentence-transformers streamlit gdown
#
# First-time setup:
#   Set GROQ_API_KEY in your environment OR enter it in the sidebar.
#   Run build_index.py once to download PDFs and build the FAISS index.
# ============================================================

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Privacy Compliance Agent",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
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
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "scenario_text" not in st.session_state:
    st.session_state.scenario_text = ""
if "do_analyze" not in st.session_state:
    st.session_state.do_analyze = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="main-header">
    <h1>🔐 AI COMPLIANCE AGENT</h1>
    <p>A RAG-based data protection and privacy risk analyst agent for university</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com/keys",
    )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### 📚 Knowledge Base")
    st.markdown(
        """
    <div class="sidebar-section" style="color: white;">
    This agent analyzes scenarios against:
    <br><br>
    📜 <b>UU PDP</b> — Personal Data Protection Law<br>
    💻 <b>UU ITE</b> — Electronic Information Law<br>
    📝 <b>UU ITE Amendment I</b><br>
    📝 <b>UU ITE Amendment II</b><br>
    🏫 <b>ALU University Regulations</b>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 🛠️ Index Management")
    if st.button("🔨 Build / Rebuild FAISS Index"):
        if not api_key:
            st.error("Enter your Groq API key first.")
        else:
            with st.spinner("Downloading PDFs and building vector index…"):
                try:
                    import gdown
                    from langchain_community.document_loaders import PyPDFLoader
                    from langchain_text_splitters import RecursiveCharacterTextSplitter

                    os.makedirs("documents", exist_ok=True)

                    documents = {
                        "UU_PDP.pdf":             "1gwzDwZZoqorirb9XXRE-acXHY9URYIrB",
                        "UU_ITE.pdf":             "1DSdMaL2cJGO4__m2CbB1MKDRDnsBb56M",
                        "UU_ITE_AmandemenI.pdf":  "1P9C3y6TG98CK9ObB_Ji-nIwP9c_2YfYw",
                        "UU_ITE_AmandemenII.pdf": "1TLL4pw8Bk1Q3BOTn7EHOqtRd2X4nsRW4",
                        "ALU_Regulations.pdf":    "1pcHYYEUYSdwYPS1Y9DrdWEbqqm0yJ1uk",
                    }
                    doc_labels = {
                        "UU_PDP.pdf":             "Personal Data Protection Law (UU PDP)",
                        "UU_ITE.pdf":             "Electronic Information & Transactions Law (UU ITE)",
                        "UU_ITE_AmandemenI.pdf":  "UU ITE Amendment I",
                        "UU_ITE_AmandemenII.pdf": "UU ITE Amendment II",
                        "ALU_Regulations.pdf":    "ALU University Regulations",
                    }

                    all_docs = []
                    for filename, file_id in documents.items():
                        path = f"documents/{filename}"
                        if not os.path.exists(path):
                            gdown.download(
                                f"https://drive.google.com/uc?id={file_id}",
                                path,
                                quiet=True,
                            )
                        loader = PyPDFLoader(path)
                        pages = loader.load()
                        for page in pages:
                            page.metadata["source_name"] = doc_labels[filename]
                            page.metadata["source_file"] = filename
                        all_docs.extend(pages)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", "Article ", "Section ", ". ", " "],
                    )
                    chunks = splitter.split_documents(all_docs)

                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vs = FAISS.from_documents(chunks, embeddings)
                    vs.save_local("faiss_index")

                    # Clear cached resource so it reloads on next use
                    st.cache_resource.clear()
                    st.success(f"✅ Index built with {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Build failed: {e}")


# ── Load LLM + vector store (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and vector store…")
def load_system(groq_api_key: str):
    os.environ["GROQ_API_KEY"] = groq_api_key

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 50},
    )

    # ── Retrieval helper ──────────────────────────────────────────────────────
    def retrieve_docs(question: str) -> str:
        docs = retriever.invoke(question)
        seen, parts, grouped = set(), [], {}
        for doc in docs:
            src_file = doc.metadata.get("source_file", "Unknown")
            src_name = doc.metadata.get("source_name", "Unknown")
            page_num = doc.metadata.get("page", "?")
            content  = doc.page_content.strip()
            key = f"{src_file}:{content[:100]}"
            if key not in seen:
                seen.add(key)
                grouped.setdefault(src_name, []).append(
                    f"  [Page {page_num}]\n  {content}"
                )
        for src_name, excerpts in grouped.items():
            parts.append(
                f"{'='*60}\n📄 SOURCE: {src_name}\n{'='*60}\n"
                + "\n\n".join(excerpts)
            )
        return "\n\n".join(parts)

    # ── Report chain ──────────────────────────────────────────────────────────
    REPORT_SYSTEM = """
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

Respond using FULL structured compliance report format:

╔══════════════════════════════════════════════════════════╗
║     🔍 DATA PROTECTION & PRIVACY RISK ANALYSIS REPORT    ║
╚══════════════════════════════════════════════════════════╝

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

If a source contains no relevant clauses, explicitly state:
"No directly applicable clauses found."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 VIOLATIONS IDENTIFIED:

[List each violation clearly, numbered]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚖️ APPLICABLE SANCTIONS:

[List applicable sanctions — criminal penalties, administrative sanctions, fines, disciplinary actions]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ RECOMMENDED ACTIONS:

Provide BOTH:
- Immediate corrective actions
- Long-term preventive measures

Use numbered actionable recommendations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENT SOURCES USED IN THIS ANALYSIS:

[List all contributing documents and page references if available]

╚══════════════════════════════════════════════════════════╝

IMPORTANT:
- Base the analysis ONLY on provided context
- Cite article numbers precisely
- Cross-reference multiple documents whenever possible
- If evidence is insufficient, clearly state limitations
"""

    report_prompt = ChatPromptTemplate.from_messages([
        ("system", REPORT_SYSTEM),
        ("human", "Legal context:\n{context}\n\nScenario:\n{question}"),
    ])

    report_chain = (
        {
            "context":  lambda x: retrieve_docs(x["question"]),
            "question": lambda x: x["question"],
        }
        | report_prompt
        | llm
        | StrOutputParser()
    )

    # ── Chat chain ────────────────────────────────────────────────────────────
    CHAT_SYSTEM = """
You are an AI Agent acting as a Data Privacy & Compliance Analyst for a university in Indonesia.

Rules:
- Use ONLY the provided legal/document context
- Do NOT invent legal references; if none exist say so explicitly
- Respond conversationally and concisely — no structured report template
- Still reference relevant laws/articles when useful
- Ask follow-up questions when appropriate
- Cross-reference multiple documents whenever possible
- Focus on university/educational data privacy compliance
"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Legal context:\n{context}\n\nQuestion:\n{question}"),
    ])

    chat_chain = (
        {
            "context":      lambda x: retrieve_docs(x["question"]),
            "question":     lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | chat_prompt
        | llm
        | StrOutputParser()
    )

    return report_chain, chat_chain


# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown("### 📥 Input Scenario")

    examples = {
        "📱 Phone Number Sale":  "A lecturer sold student phone numbers to a third-party without consent.",
        "📊 Grade Data Leak":    "IT staff accidentally published student grades and ID numbers on the public website.",
        "📷 Unauthorized CCTV":  "University installed CCTV in dorm rooms without notifying students.",
    }

    # Example buttons — write directly to the same key used by text_area
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state.scenario_text = text

    scenario = st.text_area(
        "Describe the privacy scenario:",
        height=160,
        key="scenario_text",   # reads/writes st.session_state.scenario_text
    )

    if st.button("🔍 Analyze Scenario", type="primary", use_container_width=True):
        st.session_state.do_analyze = True

with col_right:
    tab_report, tab_chat = st.tabs(["📊 Analysis Report", "💬 Discussion Chat"])

    # ── Report tab ────────────────────────────────────────────────────────────
    with tab_report:
        if st.session_state.do_analyze:
            st.session_state.do_analyze = False   # consume the trigger
            current_scenario = st.session_state.scenario_text.strip()
            if not api_key:
                st.error("Please enter your Groq API key in the sidebar.")
            elif not current_scenario:
                st.warning("Please describe a scenario first.")
            elif not os.path.exists("faiss_index"):
                st.error(
                    "FAISS index not found. Use the **Build / Rebuild FAISS Index** "
                    "button in the sidebar to create it."
                )
            else:
                with st.spinner("Analyzing scenario…"):
                    try:
                        report_chain, _ = load_system(api_key)
                        st.session_state.last_report = report_chain.invoke(
                            {"question": current_scenario}
                        )
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

        if st.session_state.last_report:
            st.markdown(
                f'<div class="result-card">{st.session_state.last_report}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download Report",
                data=st.session_state.last_report,
                file_name="compliance_report.txt",
                mime="text/plain",
            )
        else:
            st.info("Submit a scenario on the left to generate a compliance report.")

    # ── Chat tab ──────────────────────────────────────────────────────────────
    with tab_chat:
        st.caption("Ask follow-up questions or explore regulations interactively.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask anything about data protection & privacy…"):
            if not api_key:
                st.error("Please enter your Groq API key in the sidebar.")
            elif not os.path.exists("faiss_index"):
                st.error(
                    "FAISS index not found. Build it first using the sidebar button."
                )
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    _, chat_chain = load_system(api_key)

                    # Build LangChain message history from session
                    history = []
                    for m in st.session_state.messages[:-1]:
                        if m["role"] == "user":
                            history.append(HumanMessage(content=m["content"]))
                        else:
                            history.append(AIMessage(content=m["content"]))

                    with st.spinner("Thinking…"):
                        try:
                            response = chat_chain.invoke(
                                {"question": prompt, "chat_history": history}
                            )
                        except Exception as e:
                            response = f"❌ Error: {e}"

                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center; color:#808080; font-size:0.8rem;">'
    "Developed by Yunus P. | Powered by LLaMA 3.3 70B via Groq | "
    "For educational and compliance guidance purposes only. "
    "Not a substitute for qualified legal counsel."
    "</div>",
    unsafe_allow_html=True,
)
