from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vector_store


def test_retrieval(query):
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = docs = retriever.invoke(query)

    print("\n" + "=" * 80)
    print(f"QUESTION: {query}")
    print("=" * 80)

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")

        print(f"\nResult {i}")
        print(f"Source: {source}")
        print(f"Page: {page}")
        print("-" * 80)
        print(doc.page_content[:1000])  # first 1000 chars
        print("-" * 80)


if __name__ == "__main__":
    user_query = input("Enter your question: ")
    test_retrieval(user_query)