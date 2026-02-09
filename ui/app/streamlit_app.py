import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", os.environ.get("API_URL", "http://localhost:8000"))

st.set_page_config(
    page_title="Movie Night Assistant",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Night Assistant")
st.caption("Your friendly helper for planning the perfect movie night")

with st.sidebar:
    st.header("Backend Status")
    if st.button("Check Health", use_container_width=True):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if r.status_code == 200:
                st.success("✅ Backend is healthy")
            else:
                st.error(f"❌ Status: {r.status_code}")
        except requests.RequestException as e:
            st.error(f"❌ Cannot reach backend: {e}")
    
    st.divider()
    st.caption(f"Backend: {BACKEND_URL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What would you like to watch tonight?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"message": prompt},
                    timeout=60,
                )
                
                if r.status_code == 200:
                    reply = r.json().get("reply", "No response received")
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                elif r.status_code == 422:
                    error_msg = "Invalid message. Please enter some text."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})
                else:
                    error_msg = f"Error: {r.status_code} - {r.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})
            except requests.RequestException as e:
                error_msg = f"Failed to connect to backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})
