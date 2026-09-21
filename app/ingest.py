import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = r"C:\Users\yashu\OneDrive\Desktop\customer_support_assistance\data\raw"
CHROMA_PATH =r'C:\Users\yashu\OneDrive\Desktop\customer_support_knowledge_assistant\chroma_db'


def load_documents():
    documents = []

    for file_name in os.listdir(DATA_PATH):

        if file_name.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file_name)
            print(f"Loading: {file_name}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def create_vector_store(chunks, embeddings):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Chroma DB stored successfully.")


if __name__ == "__main__":
    print("Starting ingestion...")

    documents = load_documents()
    print(f"Total pages loaded: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    embeddings = create_embeddings()
    create_vector_store(chunks, embeddings)

    print("Ingestion completed successfully.")