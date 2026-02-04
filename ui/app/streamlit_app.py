import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Training Track UI", layout="centered")
st.title("Training Track UI")
st.caption(f"API_URL: {API_URL}")

st.divider()

# ---------- Health ----------
st.subheader("API status")
if st.button("Check /health"):
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code >= 400:
            st.error(f"Error: {r.status_code} - {r.text}")
        else:
            st.success("API reachable")
            st.json(r.json())
    except requests.RequestException as e:
        st.error(f"Could not reach API: {e}")

st.divider()

# ---------- Create item ----------
st.subheader("Create item")

with st.form("create_item_form"):
    text = st.text_input("Text", placeholder="Buy milk")
    is_done = st.checkbox("Done?", value=False)
    submitted = st.form_submit_button("Create")

if submitted:
    try:
        payload = {"text": text if text.strip() else None, "is_done": is_done}
        r = requests.post(f"{API_URL}/items", json=payload, timeout=5)
        if r.status_code >= 400:
            st.error(f"Error: {r.status_code} - {r.text}")
        else:
            st.success("Item created")
            st.write("Current items list:")
            st.dataframe(r.json(), use_container_width=True)
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")

st.divider()

# ---------- List items ----------
st.subheader("List items")
limit = st.number_input("Limit", min_value=1, max_value=100, value=10, step=1)

col1, col2 = st.columns(2)
with col1:
    fetch_items = st.button("Fetch /items")
with col2:
    show_raw = st.toggle("Show raw JSON", value=False)

if fetch_items:
    try:
        r = requests.get(f"{API_URL}/items", params={"limit": int(limit)}, timeout=5)
        if r.status_code >= 400:
            st.error(f"Error: {r.status_code} - {r.text}")
        else:
            data = r.json()
            if not data:
                st.info("No items yet.")
            else:
                st.dataframe(data, use_container_width=True)
                if show_raw:
                    st.json(data)
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")

st.divider()

# ---------- Get item by ID ----------
st.subheader("Get item by ID")
item_id = st.number_input("Item ID", min_value=0, value=0, step=1)

if st.button("Fetch /items/{id}"):
    try:
        r = requests.get(f"{API_URL}/items/{int(item_id)}", timeout=5)
        if r.status_code == 404:
            # API devuelve {"detail": "..."}
            try:
                st.warning(r.json().get("detail", "Not found"))
            except Exception:
                st.warning("Not found")
        elif r.status_code >= 400:
            st.error(f"Error: {r.status_code} - {r.text}")
        else:
            st.success("Item found")
            st.json(r.json())
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")

st.caption("Tip: With Docker Compose, set API_URL=http://api:8000 for the UI service.")