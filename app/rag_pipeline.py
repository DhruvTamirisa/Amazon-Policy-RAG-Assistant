from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = r"C:\Users\yashu\OneDrive\Desktop\customer_support_assistance\chroma_db"


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vector_store


def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model
# Load once when app starts
vector_store = load_vector_store()
tokenizer, model = load_llm()


def get_rag_response(query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)

    context = " ".join([doc.page_content.replace("\n", " ") for doc in docs])

    prompt = f"""
You are an AI assistant for Amazon seller and customer policy documents.

Rules:
1. Answer only from the provided context.
2. Keep the answer short, clear, and professional.
3. If the answer is not found in the context, say:
   I don't know based on the provided documents.
4. Do not make up information.
5. Give a short and clear answer in 1-2 sentences.
6. Do not repeat headings or duplicate text.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.3,
        do_sample=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    unique_sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        key = (source, page)
        if key not in seen:
            seen.add(key)
            unique_sources.append({
                "source": source,
                "page": page
            })

    return {
        "question": query,
        "answer": answer,
        "sources": unique_sources
    }




if __name__ == "__main__":
    user_query = input("Enter your question: ")
    result = get_rag_response(user_query)

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for src in result["sources"]:
        print(f"{src['source']} | Page {src['page']}")