from langchain.chains import RetrievalQA
from rag.indexing import load_faiss_index
import os

def query_faiss(query, index_path, embedding_model, llm):
    if not os.path.exists(index_path):
        raise ValueError("\u274c No transcript found. Please upload and process a file first.")
    db = load_faiss_index(index_path, embedding_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.invoke(query)