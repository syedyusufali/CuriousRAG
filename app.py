import streamlit as st
from scripts.query import RAGQueryEngine

st.set_page_config(page_title="AskMyDocs", layout="centered")
st.title("AskMyDocs: Chat With Your Files")

# Initialize the engine once in session state
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGQueryEngine()

# Text input box
question = st.text_input("Ask a question about your documents:", "What is this document about?")

# Submit button
if st.button("Submit"):
    with st.spinner("Thinking..."):
        answer = st.session_state.rag_engine.query(question)
        st.markdown("### Answer")
        st.write(answer)
