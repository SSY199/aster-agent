"""Minimal Streamlit chat UI. Talks to the FastAPI /chat endpoint —
does not call the agent graph directly, so the UI and API stay
properly decoupled (per the assignment: 'a CLI, simple web page, or
basic API is sufficient... visual polish will not affect the score').
"""

from __future__ import annotations

import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Aster & Row Support", page_icon="🎒")
st.title("Aster & Row Support")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, content, meta)

with st.sidebar:
    st.caption(f"Session: `{st.session_state.session_id[:8]}`")
    if st.button("New conversation"):
        try:
            requests.delete(f"{API_URL}/chat/{st.session_state.session_id}", timeout=5)
        except requests.RequestException:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

for role, content, meta in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)
        if meta:
            if meta.get("sources"):
                with st.expander("Sources"):
                    for s in meta["sources"]:
                        st.caption(f"📄 {s['filename']}" + (f" — {s['heading']}" if s.get("heading") else ""))
            if meta.get("handoff"):
                st.info(f"🧑‍💼 Recommending human assistance ({meta.get('handoff_reason', 'see above')})")

user_input = st.chat_input("Ask about returns, shipping, warranty, or your order...")

if user_input:
    st.session_state.history.append(("user", user_input, None))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": user_input},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                st.error(f"Could not reach the agent: {e}")
                st.stop()

        st.markdown(data["answer"])
        meta = {"sources": data["sources"], "handoff": data["handoff"], "handoff_reason": data["handoff_reason"]}
        if meta["sources"]:
            with st.expander("Sources"):
                for s in meta["sources"]:
                    st.caption(f"📄 {s['filename']}" + (f" — {s['heading']}" if s.get("heading") else ""))
        if meta["handoff"]:
            st.info(f"🧑‍💼 Recommending human assistance ({meta['handoff_reason']})")

    st.session_state.history.append(("assistant", data["answer"], meta))