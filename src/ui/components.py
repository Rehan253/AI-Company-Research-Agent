"""
Reusable Streamlit UI components.

Keeping components in a separate file makes the main app.py clean
and allows components to be reused across pages.
"""

import streamlit as st


def render_answer(answer_text: str, citations: list, confidence: float, tool_calls: int):
    """Render an agent answer with citations and metadata."""
    # Answer box
    st.markdown(answer_text)

    # Metadata row
    col1, col2, col3 = st.columns(3)
    with col1:
        color = "green" if confidence >= 0.6 else "orange" if confidence >= 0.4 else "red"
        st.markdown(f"**Confidence:** :{color}[{confidence:.0%}]")
    with col2:
        st.markdown(f"**Tool calls:** {tool_calls}")
    with col3:
        st.markdown(f"**Sources:** {len(citations)}")

    # Citations
    if citations:
        with st.expander("View Sources", expanded=False):
            for c in citations:
                page_info = f" — Page {c['page']}" if c.get("page") else ""
                st.markdown(f"**[{c['number']}]** {c['title']}{page_info}")
                st.caption(c["source"])


def render_chat_message(role: str, content: str, citations: list = None,
                        confidence: float = None, tool_calls: int = None):
    """Render a single chat message."""
    with st.chat_message(role):
        if role == "assistant" and citations is not None:
            render_answer(content, citations, confidence or 0.0, tool_calls or 0)
        else:
            st.markdown(content)


def render_company_card(name: str, chunk_count: int, on_delete=None):
    """Render a card for an ingested company."""
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.markdown(f"**{name.title()}**")
    with col2:
        st.caption(f"{chunk_count} chunks stored")
    with col3:
        if on_delete and st.button("Delete", key=f"del_{name}", type="secondary"):
            on_delete(name)


def render_ingestion_result(result: dict):
    """Render the result of a company ingestion."""
    if result.get("success"):
        st.success(f"Ingested **{result['company_name']}** successfully!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Chunks stored", result["chunks_stored"])
        col2.metric("Characters", f"{result['characters_processed']:,}")
        col3.metric("Time", f"{result['duration_seconds']}s")
        if result.get("errors"):
            st.warning(f"Minor issues: {result['errors']}")
    else:
        st.error(f"Ingestion failed: {result.get('message')}")
