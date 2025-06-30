import streamlit as st
import os
import shutil
from rag.youtube_loader import load_youtube_transcript
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.indexing import create_faiss_index
from rag.file_loader import load_file
from rag.qa import query_faiss
from models.models import embedding_model, llm

# ------------------------- Streamlit Setup -------------------------
st.set_page_config(page_title="Transcript Q&A", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "transcript_uploaded" not in st.session_state:
    st.session_state.transcript_uploaded = False
if "index_built" not in st.session_state:
    st.session_state.index_built = False

INDEX_PATH = "faiss_index"

col1, col2 = st.columns([1, 2])

# ------------------------- LEFT PANEL -------------------------
with col1:
    st.header("Upload or Import Transcript")

    source_option = st.radio("Choose input source:", ["Upload File", "YouTube URL"])

    if source_option == "Upload File":
        uploaded_file = st.file_uploader("Upload a PDF, TXT, CSV, or DOCX", type=["pdf", "txt", "csv", "docx"])
        doc_type = st.selectbox("Select document type (optional)", ["Auto", "PDF", "TXT", "CSV", "DOCX"])

        if uploaded_file and not st.session_state.index_built:
            # Clear old FAISS index if exists
            if os.path.exists(INDEX_PATH):
                try:
                    shutil.rmtree(INDEX_PATH)
                except Exception:
                    st.warning("Could not delete existing FAISS index. It may be in use.")

            st.session_state.transcript_uploaded = False

            with st.spinner("Processing file..."):
                docs = load_file(uploaded_file)
                if docs:
                    try:
                        create_faiss_index(docs, INDEX_PATH, embedding_model)
                        st.session_state.transcript_uploaded = True
                        st.session_state.index_built = True
                        st.success("File uploaded and indexed successfully!")
                    except Exception as e:
                        st.warning(f"Index creation failed. {str(e)}")
                else:
                    st.error("Unsupported file format.")

    elif source_option == "YouTube URL":
        video_url = st.text_input("Enter YouTube Video URL")

        if st.button("Fetch YouTube Transcript"):
            if video_url:
                docs, error = load_youtube_transcript(video_url)
                if docs:
                    if os.path.exists(INDEX_PATH):
                        try:
                            shutil.rmtree(INDEX_PATH)
                        except Exception:
                            st.warning("Could not delete existing FAISS index. It may be in use.")

                    try:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = splitter.split_documents(docs)
                        create_faiss_index(chunks, INDEX_PATH, embedding_model)
                        st.session_state.transcript_uploaded = True
                        st.session_state.index_built = True
                        st.success("Transcript downloaded and indexed successfully from YouTube!")
                    except Exception as e:
                        st.error(f"Error during indexing: {str(e)}")
                else:
                    st.error(f"Failed to fetch transcript: {error}")
            else:
                st.warning("Please enter a valid YouTube URL.")

# ------------------------- RIGHT PANEL -------------------------
with col2:
    st.header("Query Dashboard")
    user_query = st.text_input("Ask anything about the uploaded transcript")

    run_query = st.button("Ask")

    if run_query and user_query:
        if not st.session_state.transcript_uploaded:
            st.warning("Please upload or fetch a transcript before asking questions.")
        elif not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
            st.error("FAISS index file is missing. Please re-upload or fetch the transcript.")
            st.session_state.transcript_uploaded = False
            st.session_state.index_built = False
        else:
            try:
                with st.spinner("Searching..."):
                    answer = query_faiss(user_query, INDEX_PATH, embedding_model, llm)
                    result_text = answer["result"] if isinstance(answer, dict) and "result" in answer else str(answer)
                    st.session_state.chat_history.append(("You", user_query))
                    st.session_state.chat_history.append(("Bot", result_text))
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display conversation history
    for sender, msg in st.session_state.chat_history:
        st.markdown(f"**You:** {msg}" if sender == "You" else f"**Bot:** {msg}")
