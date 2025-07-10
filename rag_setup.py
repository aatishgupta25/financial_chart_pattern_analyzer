# rag_setup.py

import os
import torch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DOCUMENTS_DIR = 'documents'
VECTOR_STORE_PATH = 'faiss_index' # Directory to save the FAISS index


embeddings_device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def setup_rag():
    print("--- Starting RAG Setup ---")

    # 1. Load Documents
    documents = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            loader = TextLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {filename}")

    if not documents:
        print(f"No documents found in {DOCUMENTS_DIR}. Please check the directory and file extensions.")
        return

    print(f"Total documents loaded: {len(documents)}")

    # This is important for RAG to retrieve relevant small pieces
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(documents)
    print(f"Documents split into {len(splits)} chunks.")

    # Using a common sentence transformer model for embeddings
    embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print("Loading HuggingFace Embeddings model...")
    # Updated instantiation with new import and device specification
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={'device': embeddings_device})
    print("Embeddings model loaded.")

    print("Creating FAISS vector store...")
    # This step will now work after installing faiss-cpu
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("FAISS vector store created.")

    # Save the vector store to disk
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"FAISS vector store saved to {VECTOR_STORE_PATH}")
    print("--- RAG Setup Complete ---")

if __name__ == "__main__":
    setup_rag()