"""
Enterprise Document Intelligence System
Streamlit UI - Production Demo
"""
import streamlit as st
import time
from pathlib import Path

st.set_page_config(
    page_title="DocIntel · Enterprise AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #111827 !important; border-right: 1px solid #1e2d45; }
header[data-testid="stHeader"] { display: none; }
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    letter-spacing: 1px !important; transition: all 0.2s !important;
}
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: #111827 !important; border: 1px solid #1e2d45 !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-color: #00d4ff !important;
}
[data-testid="stFileUploader"] {
    background: #111827; border: 2px dashed #1e2d45; border-radius: 12px;
}
.stProgress > div > div > div {
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #111827; border-radius: 12px; padding: 4px;
    border: 1px solid #1e2d45; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; color: #64748b !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2)) !important;
    color: #e2e8f0 !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────
for key in ["rag", "agents", "doc_content", "last_result", "chunk_count", "doc_count"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 10px 0;">
        <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:#e2e8f0;">🔬 DocIntel</div>
        <div style="color:#64748b;font-size:11px;font-family:'DM Mono',monospace;margin-top:4px;">Enterprise AI · v1.0</div>
    </div>
    <hr style="border-color:#1e2d45;margin:12px 0;">
    """, unsafe_allow_html=True)

    st.markdown("**📁 Upload Documents**")
    uploaded_files = st.file_uploader(
        "Drop PDFs here", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if uploaded_files and not st.session_state.docs_loaded:
        if st.button("🚀  Process Documents", use_container_width=True):
            with st.spinner("Initialising AI system..."):
                try:
                    from sprint_bedrock import get_llm, get_embeddings
                    from sprint_rag import SimpleRAG
                    from sprint_agents import MultiAgentSystem

                    save_dir = Path("./data/sample_contracts")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    for f in uploaded_files:
                        with open(save_dir / f.name, "wb") as out:
                            out.write(f.getbuffer())

                    llm = get_llm()
                    embeddings = get_embeddings()
                    rag = SimpleRAG(llm, embeddings)
                    docs = rag.load_documents(str(save_dir))
                    chunks = rag.create_vector_store(docs)
                    rag.setup_qa_chain()

                    st.session_state.rag = rag
                    st.session_state.agents = MultiAgentSystem(llm, rag)
                    st.session_state.docs_loaded = True
                    st.session_state.doc_count = len(uploaded_files)
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.doc_content = "\n".join(
                        [d.page_content[:600] for d in docs[:6]]
                    )
                    st.success(f"✅ {len(uploaded_files)} docs loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    elif st.session_state.docs_loaded:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.1);border:1px solid #10b981;border-radius:10px;padding:14px 16px;margin-bottom:12px;">
            <div style="color:#10b981;font-family:'Syne',sans-serif;font-weight:700;font-size:12px;margin-bottom:6px;">✅ SYSTEM READY</div>
            <div style="color:#64748b;font-size:11px;font-family:'DM Mono',monospace;">
                {st.session_state.doc_count} documents loaded<br>
                {st.session_state.chunk_count} searchable chunks<br>
                Claude 3 · Titan Embeddings · FAISS
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄  Reset", use_container_width=True):
            for k in ["rag","agents","docs_loaded","doc_content","last_result"]:
                st.session_state[k] = None
            st.session_state.docs_loaded = False
            st.rerun()

    st.markdown("""
    <hr style="border-color:#1e2d45;margin:20px 0 12px 0;">
    <div style="color:#64748b;font-size:11px;font-family:'DM Mono',monospace;line-height:2.2;">
        🟡 Amazon Bedrock<br>🔵 Claude 3 Haiku<br>🟢 Titan Embeddings<br>
        🟣 LangGraph<br>🔴 FAISS Vector DB<br>⚪ Streamlit UI
    </div>
    """, unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 20px 0;">
    <div style="display:inline-block;background:linear-gradient(135deg,rgba(0,212,255,0.1),rgba(124,58,237,0.1));
        border:1px solid rgba(0,212,255,0.3);color:#00d4ff;font-family:'DM Mono',monospace;
        font-size:10px;letter-spacing:2px;text-transform:uppercase;padding:5px 14px;
        border-radius:20px;margin-bottom:16px;">
        Amazon Bedrock · LangGraph · Multi-Agent RAG
    </div>
    <h1 style="font-family:'Syne',sans-serif;font-size:40px;font-weight:800;
        background:linear-gradient(135deg,#fff 0%,#00d4ff 50%,#7c3aed 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;margin:0 0 8px 0;line-height:1.1;">
        Enterprise Document<br>Intelligence System
    </h1>
    <p style="color:#64748b;font-size:15px;margin:0;font-weight:300;">
        Multi-agent AI · Contract Analysis · Risk Detection · Built with Amazon Bedrock
    </p>
</div>
""", unsafe_allow_html=True)

# ── STATS ROW ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
stats = [
    (str(st.session_state.doc_count or "—"), "Documents", "#00d4ff"),
    (str(st.session_state.chunk_count or "—"), "Chunks Indexed", "#7c3aed"),
    (str(len(st.session_state.last_result["messages"]) if st.session_state.last_result else "—"), "Agent Calls", "#10b981"),
    ("£0.04", "Cost Per Analysis", "#f59e0b"),
]
for col, (val, label, color) in zip([c1,c2,c3,c4], stats):
    with col:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:20px;text-align:center;">
            <div style="font-family:'Syne',sans-serif;font-size:30px;font-weight:800;color:{color};line-height:1;margin-bottom:8px;">{val}</div>
            <div style="color:#64748b;font-size:12px;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── NOT LOADED ────────────────────────────────────────────────────────────
if not st.session_state.docs_loaded:
    st.markdown("""
    <div style="background:#111827;border:2px dashed #1e2d45;border-radius:16px;
        padding:60px 40px;text-align:center;margin:20px 0;">
        <div style="font-size:48px;margin-bottom:16px;">📄</div>
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;
            color:#e2e8f0;margin-bottom:8px;">Upload contracts to begin</div>
        <div style="color:#64748b;font-size:14px;max-width:400px;margin:0 auto;line-height:1.6;">
            Drop PDF files in the sidebar, then click <strong style="color:#00d4ff">Process Documents</strong>
        </div>
        <div style="margin-top:24px;display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
            <div style="background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.2);
                border-radius:8px;padding:8px 14px;font-size:12px;color:#00d4ff;font-family:'DM Mono',monospace;">
                💬 Q&A with citations</div>
            <div style="background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.2);
                border-radius:8px;padding:8px 14px;font-size:12px;color:#a78bfa;font-family:'DM Mono',monospace;">
                ⚠️ Risk detection</div>
            <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.2);
                border-radius:8px;padding:8px 14px;font-size:12px;color:#10b981;font-family:'DM Mono',monospace;">
                📋 Auto summary</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── LOADED: TABS ──────────────────────────────────────────────────────────
else:
    tab1, tab2, tab3 = st.tabs(["💬  Ask a Question", "🤖  Full Agent Analysis", "📊  Last Report"])

    # TAB 1: Q&A
    with tab1:
        st.markdown("""
        <div style="margin-bottom:20px;">
            <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;
                color:#e2e8f0;margin-bottom:6px;">RAG Question & Answer</div>
            <div style="color:#64748b;font-size:14px;">Ask anything. The AI searches all documents and answers with exact source citations.</div>
        </div>
        """, unsafe_allow_html=True)

        question = st.text_input("", placeholder="e.g. What are the payment terms? / Which contract has the highest risk?", label_visibility="collapsed")
        ask_btn = st.button("🔍  Search Documents")

        if ask_btn and question:
            with st.spinner("Searching documents..."):
                t0 = time.time()
                result = st.session_state.rag.ask(question)
                elapsed = time.time() - t0

            st.markdown(f"""
            <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);
                border-radius:12px;padding:24px;margin:16px 0;">
                <div style="color:#64748b;font-size:11px;font-family:'DM Mono',monospace;
                    margin-bottom:10px;letter-spacing:1px;">ANSWER · {elapsed:.1f}s · Claude 3 + FAISS</div>
                <div style="font-size:15px;color:#e2e8f0;line-height:1.8;">{result['answer']}</div>
            </div>
            """, unsafe_allow_html=True)

            sources = list(set(result["sources"]))
            src_html = "".join([
                f'<div style="font-family:DM Mono,monospace;font-size:12px;color:#64748b;padding:5px 0;border-top:1px solid #1e2d45;">📄 {Path(s).name}</div>'
                for s in sources
            ])
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:16px 20px;margin-top:8px;">
                <div style="font-family:'Syne',sans-serif;font-size:11px;font-weight:700;
                    letter-spacing:2px;color:#00d4ff;text-transform:uppercase;margin-bottom:8px;">Sources</div>
                {src_html}
            </div>
            """, unsafe_allow_html=True)
        elif ask_btn:
            st.warning("Please enter a question.")

    # TAB 2: MULTI-AGENT
    with tab2:
        st.markdown("""
        <div style="margin-bottom:20px;">
            <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;
                color:#e2e8f0;margin-bottom:6px;">Multi-Agent Workflow</div>
            <div style="color:#64748b;font-size:14px;">4 AI agents coordinate: Router → RAG → Summarizer → Risk Analyzer → Finalizer</div>
        </div>
        """, unsafe_allow_html=True)

        analysis_q = st.text_input(
            "", value="Which contract has the highest risk and why?", label_visibility="collapsed"
        )
        run_btn = st.button("🤖  Run Multi-Agent Analysis")

        if run_btn:
            agents_steps = [
                ("🎯", "ROUTER", "Analysing task and routing to first agent...", "#00d4ff", 10),
                ("💬", "RAG AGENT", "Searching document chunks for answer...", "#10b981", 30),
                ("🎯", "ROUTER", "RAG complete — routing to summarizer...", "#00d4ff", 45),
                ("📋", "SUMMARIZER", "Creating executive summary of all contracts...", "#7c3aed", 65),
                ("🎯", "ROUTER", "Summary done — routing to risk analyzer...", "#00d4ff", 75),
                ("⚠️", "RISK ANALYZER", "Scanning all contracts for risk clauses...", "#f59e0b", 90),
                ("📊", "FINALIZER", "Compiling full intelligence report...", "#ef4444", 100),
            ]

            progress_bar = st.progress(0)
            status_box = st.empty()

            for icon, name, desc, color, pct in agents_steps:
                progress_bar.progress(pct)
                status_box.markdown(f"""
                <div style="background:rgba(0,0,0,0.3);border:1px solid {color}44;
                    border-radius:10px;padding:14px 20px;font-family:'DM Mono',monospace;
                    font-size:13px;color:{color};">
                    {icon} <strong>{name}</strong> — {desc}
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.5)

            with st.spinner(""):
                t0 = time.time()
                result = st.session_state.agents.run(
                    question=analysis_q,
                    doc_content=st.session_state.doc_content
                )
                elapsed = time.time() - t0

            st.session_state.last_result = result
            status_box.markdown(f"""
            <div style="background:rgba(16,185,129,0.1);border:1px solid #10b981;
                border-radius:10px;padding:14px 20px;font-family:'DM Mono',monospace;
                font-size:13px;color:#10b981;">
                ✅ <strong>WORKFLOW COMPLETE</strong> — {len(result['messages'])} agent calls · {elapsed:.1f}s
                <span style="color:#64748b;margin-left:16px;">→ View full report in the Report tab</span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Agent Workflow Trace"):
                for msg in result.get("messages", []):
                    st.markdown(f"""
                    <div style="font-family:'DM Mono',monospace;font-size:12px;color:#64748b;
                        padding:5px 0;border-bottom:1px solid #1e2d45;">→ {msg.content}</div>
                    """, unsafe_allow_html=True)

    # TAB 3: REPORT
    with tab3:
        if not st.session_state.last_result:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#64748b;">
                <div style="font-size:40px;margin-bottom:16px;">📊</div>
                <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;
                    color:#e2e8f0;margin-bottom:8px;">No report yet</div>
                <div style="font-size:14px;">Run a Multi-Agent Analysis to generate your report</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            result = st.session_state.last_result
            report = result.get("final_output", "")

            # Render beautifully
            st.markdown("""
            <div style="background:linear-gradient(135deg,rgba(0,212,255,0.04),rgba(124,58,237,0.04));
                border:1px solid #1e2d45;border-radius:16px;padding:28px 32px;">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;
                    padding-bottom:16px;border-bottom:1px solid #1e2d45;">
                    <div style="width:10px;height:10px;border-radius:50%;background:#10b981;
                        box-shadow:0 0 8px #10b981;"></div>
                    <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                        color:#e2e8f0;letter-spacing:1px;">DOCUMENT INTELLIGENCE REPORT</div>
                </div>
            """, unsafe_allow_html=True)

            for line in report.split("\n"):
                line = line.strip()
                if not line or line.startswith("# "):
                    continue
                elif line.startswith("## "):
                    title = line[3:].strip()
                    st.markdown(f"""
                    <div style="font-family:'Syne',sans-serif;font-size:12px;font-weight:700;
                        letter-spacing:2px;text-transform:uppercase;color:#00d4ff;
                        margin:20px 0 10px 0;padding-bottom:8px;border-bottom:1px solid #1e2d45;">{title}</div>
                    """, unsafe_allow_html=True)
                elif line.startswith("**Q:**"):
                    st.markdown(f"""
                    <div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
                        border-radius:8px;padding:14px 18px;margin:8px 0;">
                        <div style="color:#64748b;font-size:12px;margin-bottom:6px;">QUESTION</div>
                        <div style="color:#e2e8f0;font-size:14px;">{line[7:].strip()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif line.startswith("**A:**"):
                    st.markdown(f"""
                    <div style="background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.15);
                        border-radius:8px;padding:14px 18px;margin:8px 0;">
                        <div style="color:#10b981;font-size:12px;margin-bottom:6px;">ANSWER</div>
                        <div style="color:#e2e8f0;font-size:14px;line-height:1.7;">{line[6:].strip()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif line.startswith("---"):
                    st.markdown('<hr style="border-color:#1e2d45;margin:20px 0;">', unsafe_allow_html=True)
                elif line.startswith("*") and line.endswith("*"):
                    st.markdown(f'<div style="color:#64748b;font-size:12px;font-family:DM Mono,monospace;margin-top:8px;">{line[1:-1]}</div>', unsafe_allow_html=True)
                else:
                    # Highlight risk levels
                    styled = line
                    risk_colors = {"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "MEDIUM": "#63b3ed", "LOW": "#10b981"}
                    for lvl, clr in risk_colors.items():
                        styled = styled.replace(f"({lvl})", f'<span style="color:{clr};font-family:DM Mono,monospace;font-size:12px;font-weight:700;">({lvl})</span>')
                        styled = styled.replace(f"[{lvl}]", f'<span style="color:{clr};font-family:DM Mono,monospace;font-size:12px;font-weight:700;">[{lvl}]</span>')
                    st.markdown(f'<div style="color:#94a3b8;font-size:14px;line-height:1.8;padding:3px 0;">{styled}</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.download_button(
                "📥  Download Report",
                data=report,
                file_name="document_intelligence_report.md",
                mime="text/markdown"
            )
