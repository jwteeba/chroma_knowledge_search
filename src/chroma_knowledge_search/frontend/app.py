import os
import streamlit as st
import requests

st.set_page_config(page_title="Chroma Knowledge Search", layout="centered")
st.title("Chroma Knowledge Search")

API_BASE_DEFAULT = os.getenv("API_BASE", "http://localhost:8000/api")

if "api_base" not in st.session_state:
    st.session_state["api_base"] = API_BASE_DEFAULT
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

with st.sidebar:
    st.text_input("API Base", key="api_base")
    st.text_input("API Key", key="api_key", type="password")
    st.caption("Set API key in your .env as API_KEY")

if not st.session_state["api_key"]:
    st.warning("Enter API key to continue.")
    st.stop()

headers = {"x-api-key": st.session_state["api_key"]}

st.header("1) Upload documents")
uploaded = st.file_uploader("Upload (pdf/txt/docx)", type=["pdf", "txt", "docx"])
if uploaded is not None and st.button("Upload"):
    files = {"file": (uploaded.name, uploaded.getvalue())}
    try:
        r = requests.post(
            f"{st.session_state['api_base']}/upload",
            files=files,
            headers=headers,
            timeout=60,
        )
        if r.status_code == 200:
            st.success("Uploaded")
            st.json(r.json())
        else:
            st.error(r.text)
    except Exception as e:
        st.error(f"Upload failed: {e}")

st.header("2) Ask questions")
q = st.text_input("Your question")
top_k = st.slider("Top-K", 1, 10, 5)
if st.button("Ask") and q.strip():
    try:
        r = requests.post(
            f"{st.session_state['api_base']}/query",
            json={"query": q, "top_k": top_k},
            headers=headers,
            timeout=60,
        )
        if r.status_code == 200:
            res = r.json()
            st.subheader("Answer")
            st.write(res["answer"])
            st.subheader("Sources")
            st.write(res["sources"])
        else:
            st.error(r.text)
    except Exception as e:
        st.error(f"Query failed: {e}")
