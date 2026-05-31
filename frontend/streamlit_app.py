import streamlit as st

st.set_page_config(page_title="AI Data Extraction", layout="wide")

st.title("AI Data Extraction")
st.write("Upload a document to extract structured data.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.info("Processing not yet implemented.")
