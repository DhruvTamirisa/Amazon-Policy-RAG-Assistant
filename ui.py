import streamlit as st
import requests

st.title("Amazon Policy RAG Assistant")

question = st.text_input("Ask a question")

if st.button("Get Answer"):
    if question.strip():
        try:
            with st.spinner("Generating answer..."):
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": question},
                    timeout=120
                )

            st.write("Status code:", response.status_code)

            if response.status_code == 200:
                result = response.json()

                st.subheader("Answer")
                st.write(result.get("answer", "No answer returned"))

                st.subheader("Sources")
                for src in result.get("sources", []):
                    st.write(f"{src['source']} | Page {src['page']}")
            else:
                st.error(f"Error while fetching answer. Status code: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.Timeout:
            st.error("The request timed out. The backend may still be generating the answer.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI backend. Make sure it is running on port 8000.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")