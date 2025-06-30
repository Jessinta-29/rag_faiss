import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def on_rm_error(func, path, exc_info):
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)

def create_faiss_index(docs, index_path, embedding_model):
    if not docs:
        return False
    try:
        if os.path.exists(index_path):
            shutil.rmtree(index_path, onerror=on_rm_error)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(index_path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create FAISS index: {e}")
        return False

def load_faiss_index(index_path, embedding_model):
    faiss_file = os.path.join(index_path, "index.faiss")
    if not os.path.exists(faiss_file):
        raise FileNotFoundError(f"FAISS file not found at {faiss_file}")
    return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
