"""
AI Company Research Agent — Professional Streamlit UI
"""

import requests
import streamlit as st

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Company Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8080/api/v1"

# ─────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --primary:    #6C63FF;
    --primary-dark: #5A52E0;
    --accent:     #00D4AA;
    --danger:     #FF4B6E;
    --bg-card:    #1A1D2E;
    --bg-input:   #0E1117;
    --border:     rgba(108,99,255,0.25);
    --text-main:  #FAFAFA;
    --text-muted: #9AA3BF;
}

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1100px; }

/* ── Force dark background everywhere ── */
.stApp { background-color: #0E1117 !important; }
section[data-testid="stMain"] { background-color: #0E1117 !important; }
.main { background-color: #0E1117 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0F1C 0%, #12152B 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1A1D2E 0%, #12152B 50%, #1A1D2E 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(108,99,255,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(0,212,170,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #FFFFFF, var(--primary));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}
.hero p { color: var(--text-muted); font-size: 1rem; margin: 0; }

/* ── Section title ── */
.section-title {
    font-size: 1rem; font-weight: 600; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 1.5rem 0 0.75rem;
}

/* ── Glassmorphism card ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.glass-card:hover { border-color: var(--primary); }

/* ── Metric chips ── */
.chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.5rem 0; }
.chip {
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem; font-weight: 500;
    color: #A89DFF;
}
.chip.green {
    background: rgba(0,212,170,0.1);
    border-color: rgba(0,212,170,0.3);
    color: #00D4AA;
}
.chip.red {
    background: rgba(255,75,110,0.1);
    border-color: rgba(255,75,110,0.3);
    color: #FF4B6E;
}

/* ── Chat bubbles ── */
.chat-wrap { display: flex; flex-direction: column; gap: 1rem; padding: 0.5rem 0; }

.bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.2rem;
    max-width: 75%;
    font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3);
}

.bubble-ai {
    align-self: flex-start;
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-main);
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.25rem;
    max-width: 85%;
    font-size: 0.93rem;
    line-height: 1.65;
}
.bubble-ai strong { color: #A89DFF; }

.bubble-meta {
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    margin-top: 0.6rem; padding-top: 0.6rem;
    border-top: 1px solid rgba(255,255,255,0.07);
    font-size: 0.75rem; color: var(--text-muted);
}

/* ── Source citation block ── */
.citation {
    background: rgba(0,212,170,0.06);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.4rem 0.75rem;
    margin-top: 0.4rem;
    font-size: 0.8rem;
    color: var(--text-muted);
}
.citation a { color: var(--accent); text-decoration: none; }

/* ── Suggestion buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1E2240, #252847) !important;
    border: 1px solid rgba(108,99,255,0.5) !important;
    border-radius: 12px !important;
    color: #E0DDFF !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    text-align: left !important;
    line-height: 1.4 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2A2D5A, #323580) !important;
    border-color: #6C63FF !important;
    color: #fff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
}

/* ── Primary action button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF, #5A52E0) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(108,99,255,0.5) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #7B73FF, #6C63FF) !important;
    box-shadow: 0 6px 20px rgba(108,99,255,0.65) !important;
    transform: translateY(-2px) !important;
}

/* ── Secondary button (delete/clear) ── */
.stButton > button[kind="secondary"] {
    background: rgba(255,75,110,0.1) !important;
    border: 1px solid rgba(255,75,110,0.35) !important;
    color: #FF8099 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,75,110,0.2) !important;
    border-color: #FF4B6E !important;
    color: #fff !important;
}

/* ── Inputs ── */
.stTextInput input, div[data-baseweb="select"] {
    background: #12152B !important;
    border: 1px solid rgba(108,99,255,0.3) !important;
    border-radius: 10px !important;
    color: #FAFAFA !important;
}
.stTextInput input:focus { border-color: #6C63FF !important; box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important; }
.stTextInput input::placeholder { color: #5A6280 !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #12152B !important;
    border: 2px solid rgba(108,99,255,0.4) !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #FAFAFA !important;
    font-size: 0.95rem !important;
    caret-color: #6C63FF !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #5A6280 !important;
    font-style: italic !important;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.85rem;
    font-size: 0.78rem; color: #00D4AA; font-weight: 500;
}
.status-badge.offline {
    background: rgba(255,75,110,0.1);
    border-color: rgba(255,75,110,0.3);
    color: #FF4B6E;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

/* ── Company list card ── */
.company-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.65rem;
    display: flex; align-items: center; justify-content: space-between;
    transition: border-color 0.2s;
}
.company-card:hover { border-color: var(--primary); }
.company-name { font-weight: 600; font-size: 1rem; color: var(--text-main); }
.company-meta { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem; }

/* ── Divider ── */
.custom-divider {
    border: none; border-top: 1px solid var(--border);
    margin: 1.25rem 0;
}

/* ── Spinner override ── */
.stSpinner > div { border-top-color: var(--primary) !important; }

/* ── Expander ── */
details summary {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}

/* ── Selectbox label ── */
label { color: var(--text-muted) !important; font-size: 0.85rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────

def api_get(path):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path, body, timeout=120):
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def api_delete(path):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def check_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def confidence_color(score):
    if score >= 0.65:
        return "green"
    if score >= 0.4:
        return "#FFA500"
    return "red"


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

for key, default in [
    ("chat_history", []),
    ("current_company", ""),
    ("pending_q", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1rem'>
        <div style='font-size:1.5rem;font-weight:700;color:#FAFAFA'>🔍 ResearchAI</div>
        <div style='font-size:0.78rem;color:#9AA3BF;margin-top:0.2rem'>Company Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    api_alive = check_health()
    badge_class = "status-badge" if api_alive else "status-badge offline"
    badge_text = "API Online" if api_alive else "API Offline"
    st.markdown(
        f'<div class="{badge_class}"><div class="dot"></div>{badge_text}</div>',
        unsafe_allow_html=True,
    )

    if not api_alive:
        st.markdown("""
        <div style='margin-top:0.75rem;padding:0.75rem;background:rgba(255,75,110,0.08);
                    border:1px solid rgba(255,75,110,0.2);border-radius:10px;
                    font-size:0.8rem;color:#FF8099'>
            Start the backend first:<br>
            <code style='color:#FFB3C1'>uvicorn src.api.main:app --port 8000</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🔬  Research", "🏢  Companies", "ℹ️  About"],
        label_visibility="collapsed",
    )

    # Quick company switcher
    st.markdown("<div class='section-title'>Active Company</div>", unsafe_allow_html=True)
    data = api_get("/companies")
    companies = [c["name"].title() for c in (data or {}).get("companies", [])]

    if companies:
        sel = st.selectbox("", companies, label_visibility="collapsed",
                           key="sidebar_company",
                           index=companies.index(st.session_state.current_company.title())
                           if st.session_state.current_company.title() in companies else 0)
        if sel != st.session_state.current_company:
            st.session_state.current_company = sel
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.caption("No companies yet.")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem;color:#9AA3BF'>
        Built by <strong style='color:#C5C0FF'>Rehan Shafique</strong><br>
        AI Engineer Portfolio · France 🇫🇷
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: RESEARCH
# ─────────────────────────────────────────────

if "Research" in page:

    # Hero
    st.markdown("""
    <div class='hero'>
        <h1>Company Research Agent</h1>
        <p>Ask any question about a company — the AI finds, retrieves, and cites every answer.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Ingest panel ──
    with st.expander("➕  Add a new company to the knowledge base", expanded=not st.session_state.current_company):
        c1, c2 = st.columns([3, 1])
        with c1:
            new_co = st.text_input("Company name", placeholder="e.g. Airbus, BNP Paribas, Renault")
            co_url = st.text_input("Official website (optional)", placeholder="https://www.airbus.com")
        with c2:
            st.write("")
            st.write("")
            st.write("")
            do_ingest = st.button("🚀 Research Company", type="primary", use_container_width=True)

        if do_ingest and new_co.strip():
            with st.spinner(f"Ingesting **{new_co}** — scraping Wikipedia + website…"):
                res = api_post("/ingest", {
                    "company_name": new_co.strip(),
                    "company_url": co_url.strip() or None,
                    "replace_existing": False,
                }, timeout=90)

            if res and res.get("success"):
                st.markdown(f"""
                <div class='glass-card' style='border-color:rgba(0,212,170,0.4)'>
                    <div style='color:#00D4AA;font-weight:600;font-size:1rem'>
                        ✅ {res['company_name']} ingested successfully!
                    </div>
                    <div class='chip-row' style='margin-top:0.6rem'>
                        <span class='chip green'>📦 {res['chunks_stored']} chunks</span>
                        <span class='chip green'>📄 {res['characters_processed']:,} characters</span>
                        <span class='chip green'>⏱ {res['duration_seconds']}s</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.current_company = new_co.strip()
                st.session_state.chat_history = []
                st.rerun()
            elif res and res.get("error"):
                st.error(f"Error: {res['error']}")
            elif res:
                st.warning(f"Partial ingestion: {res.get('message')}")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Company selector ──
    if not companies:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:2.5rem'>
            <div style='font-size:2.5rem'>🏢</div>
            <div style='font-size:1.1rem;font-weight:600;color:#FAFAFA;margin:0.5rem 0'>No companies yet</div>
            <div style='color:#9AA3BF;font-size:0.9rem'>Use the panel above to ingest your first company.</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    company = st.session_state.current_company or companies[0]
    if not st.session_state.current_company:
        st.session_state.current_company = company

    # Company header pill
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem'>
        <div style='width:38px;height:38px;background:linear-gradient(135deg,#6C63FF,#00D4AA);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;
                    font-size:1.1rem'>🏢</div>
        <div>
            <div style='font-size:1.15rem;font-weight:700;color:#FAFAFA'>{company}</div>
            <div style='font-size:0.78rem;color:#9AA3BF'>Active research target</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Suggested questions ──
    st.markdown("<div class='section-title'>Suggested Questions</div>", unsafe_allow_html=True)
    suggestions = [
        f"Who founded {company} and when?",
        f"What are {company}'s main products or services?",
        f"What is {company}'s annual revenue?",
        f"What is the latest news about {company}?",
        f"Who is the CEO of {company}?",
        f"What are {company}'s main competitive advantages?",
    ]
    cols = st.columns(3)
    for i, sug in enumerate(suggestions):
        if cols[i % 3].button(sug, key=f"sug_{i}"):
            st.session_state.pending_q = sug

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Chat history ──
    st.markdown("<div class='section-title'>Conversation</div>", unsafe_allow_html=True)

    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(
                        f"<div style='color:#E0DDFF;font-size:0.95rem'>{msg['content']}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                with st.chat_message("assistant"):
                    conf       = msg.get("confidence", 0)
                    tool_n     = msg.get("tool_calls", 0)
                    citations  = msg.get("citations", [])
                    conf_color = confidence_color(conf)

                    # Render the answer text as markdown (handles [1][2] and Sources: properly)
                    st.markdown(msg["content"])

                    # Citations as styled links
                    if citations:
                        cite_html = ""
                        for c in citations[:5]:
                            src   = c.get("source", "")
                            title = c.get("title", src)[:60]
                            cite_html += (
                                f'<div class="citation">'
                                f'[{c["number"]}] <a href="{src}" target="_blank">{title}</a>'
                                f'</div>'
                            )
                        st.markdown(cite_html, unsafe_allow_html=True)

                    # Metadata row
                    st.markdown(
                        f"""<div class='bubble-meta'>
                            <span>🎯 Confidence: <strong style='color:{conf_color}'>{conf:.0%}</strong></span>
                            <span>🔧 Tool calls: <strong>{tool_n}</strong></span>
                            <span>📚 Sources cited: <strong>{len(citations)}</strong></span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        st.write("")
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown("""
        <div style='text-align:center;padding:2rem;color:#9AA3BF;
                    border:1px dashed rgba(108,99,255,0.2);border-radius:14px'>
            <div style='font-size:1.8rem'>💡</div>
            <div style='margin-top:0.4rem;font-size:0.9rem'>
                Click a suggested question above or type below to start
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat input ──
    pending = st.session_state.pop("pending_q", "") if "pending_q" in st.session_state else ""
    user_input = st.chat_input(f"Search or ask anything about {company} — e.g. 'What is their revenue?' or 'Who is the CEO?'")
    question = pending or user_input

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("🤖 Agent is researching… (using tools autonomously)"):
            result = api_post("/query", {
                "company_name": company,
                "question": question,
            }, timeout=120)

        if result and not result.get("error"):
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result.get("answer", "No answer returned."),
                "citations": result.get("citations", []),
                "confidence": result.get("confidence", 0.0),
                "tool_calls": result.get("tool_calls_made", 0),
            })
        elif result and result.get("error") == "timeout":
            st.error("The agent timed out. Try a simpler question.")
        else:
            st.error("Something went wrong. Is the API running?")

        st.rerun()


# ─────────────────────────────────────────────
# PAGE: COMPANIES
# ─────────────────────────────────────────────

elif "Companies" in page:

    st.markdown("""
    <div class='hero'>
        <h1>Knowledge Base</h1>
        <p>All companies stored in ChromaDB — ready for instant Q&amp;A.</p>
    </div>
    """, unsafe_allow_html=True)

    data = api_get("/companies")
    if not data or not data["companies"]:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:2.5rem'>
            <div style='font-size:2.5rem'>📭</div>
            <div style='font-size:1.1rem;font-weight:600;color:#FAFAFA;margin:0.5rem 0'>
                Knowledge base is empty
            </div>
            <div style='color:#9AA3BF'>Go to Research to ingest your first company.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        total = data["total"]
        total_chunks = sum(c["chunk_count"] for c in data["companies"])

        m1, m2 = st.columns(2)
        m1.markdown(f"""
        <div class='glass-card' style='text-align:center'>
            <div style='font-size:2rem;font-weight:700;color:#6C63FF'>{total}</div>
            <div style='color:#9AA3BF;font-size:0.85rem'>Companies Ingested</div>
        </div>
        """, unsafe_allow_html=True)
        m2.markdown(f"""
        <div class='glass-card' style='text-align:center'>
            <div style='font-size:2rem;font-weight:700;color:#00D4AA'>{total_chunks:,}</div>
            <div style='color:#9AA3BF;font-size:0.85rem'>Total Chunks Stored</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Ingested Companies</div>", unsafe_allow_html=True)

        for co in data["companies"]:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div class='glass-card'>
                    <div style='display:flex;align-items:center;gap:0.75rem'>
                        <div style='width:36px;height:36px;background:linear-gradient(135deg,#6C63FF,#5A52E0);
                                    border-radius:9px;display:flex;align-items:center;justify-content:center;
                                    font-size:1rem'>🏢</div>
                        <div>
                            <div class='company-name'>{co['name'].title()}</div>
                            <div class='company-meta'>{co['chunk_count']} chunks · ChromaDB collection: <code>{co['name']}</code></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.write("")
                if st.button("🗑️ Delete", key=f"del_{co['name']}", type="secondary"):
                    if api_delete(f"/companies/{co['name']}"):
                        st.success(f"Deleted {co['name']}")
                        st.rerun()


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────

elif "About" in page:

    st.markdown("""
    <div class='hero'>
        <h1>About This Project</h1>
        <p>A production-grade AI research agent built as a portfolio project for AI Engineer alternance in France.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='glass-card'>
            <div style='font-size:1rem;font-weight:600;color:#FAFAFA;margin-bottom:0.75rem'>
                🧠 How it works
            </div>
            <div style='color:#9AA3BF;font-size:0.88rem;line-height:1.75'>
                <strong style='color:#C5C0FF'>1. Ingest</strong> — Scrapes Wikipedia + website, chunks text, generates vector embeddings, stores in ChromaDB.<br><br>
                <strong style='color:#C5C0FF'>2. Retrieve</strong> — Uses MMR semantic search to find relevant chunks without redundancy.<br><br>
                <strong style='color:#C5C0FF'>3. Generate</strong> — GPT-4o-mini synthesizes a grounded answer with numbered citations.<br><br>
                <strong style='color:#C5C0FF'>4. Agent</strong> — LangGraph ReAct agent autonomously decides: vector search vs. web search vs. summarize.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='glass-card' style='margin-top:0.75rem'>
            <div style='font-size:1rem;font-weight:600;color:#FAFAFA;margin-bottom:0.75rem'>
                🗂️ Architecture
            </div>
            <pre style='color:#A89DFF;font-size:0.78rem;background:rgba(0,0,0,0.2);
                        padding:0.75rem;border-radius:8px;overflow-x:auto'>
User Question
    ↓
LangGraph Agent (ReAct loop)
    ├── vector_search  → ChromaDB
    ├── web_search     → Tavily API
    ├── summarize_text → GPT-4o-mini
    └── ingest_company → Pipeline
    ↓
Cited Answer with Sources</pre>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-card'>
            <div style='font-size:1rem;font-weight:600;color:#FAFAFA;margin-bottom:0.75rem'>
                ⚙️ Tech Stack
            </div>
        """, unsafe_allow_html=True)

        stack = [
            ("🔗", "LangChain 0.3+", "LLM framework"),
            ("🕸️", "LangGraph", "Agent orchestration (ReAct)"),
            ("🗄️", "ChromaDB", "Vector database (local)"),
            ("🧬", "sentence-transformers", "Free local embeddings (384d)"),
            ("🤖", "GPT-4o-mini", "LLM for generation"),
            ("🔍", "Tavily API", "Live web search"),
            ("⚡", "FastAPI", "Async REST API backend"),
            ("🎨", "Streamlit", "Frontend UI"),
            ("📄", "PyMuPDF", "PDF text extraction"),
        ]
        for icon, name, desc in stack:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.75rem;
                        padding:0.45rem 0;border-bottom:1px solid rgba(255,255,255,0.05)'>
                <span style='font-size:1rem'>{icon}</span>
                <div>
                    <span style='color:#FAFAFA;font-weight:500;font-size:0.88rem'>{name}</span>
                    <span style='color:#9AA3BF;font-size:0.8rem'> — {desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='glass-card' style='margin-top:0.75rem;
                    background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(0,212,170,0.08))'>
            <div style='font-size:1rem;font-weight:600;color:#FAFAFA;margin-bottom:0.5rem'>
                👤 Built by
            </div>
            <div style='font-size:0.95rem;font-weight:600;color:#C5C0FF'>Rehan Shafique</div>
            <div style='color:#9AA3BF;font-size:0.83rem;margin-top:0.25rem'>
                AI Engineer Alternance Portfolio · France 🇫🇷<br>
                RAG · LangGraph · Vector Search · FastAPI
            </div>
        </div>
        """, unsafe_allow_html=True)
