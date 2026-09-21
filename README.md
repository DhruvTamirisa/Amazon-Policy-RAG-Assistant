# Amazon Policy RAG Assistant

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) system that answers questions from Amazon seller and customer policy documents.

## Domain
Amazon E-commerce Seller Policies & Customer Support Policies

## Features
- Reads multiple Amazon policy PDFs
- Converts text into embeddings
- Stores vectors in ChromaDB
- Retrieves relevant document chunks
- Uses HuggingFace LLM to generate answers
- Displays source file names and page numbers
- Interactive Streamlit interface

## Tech Stack
Python  
LangChain  
ChromaDB  
Sentence Transformers  
HuggingFace Transformers  
FastAPI  
Streamlit  

## Documents Used
- Amazon Seller Code of Conduct
- Amazon Services Business Solutions Agreement
- Amazon Seller Guidelines
- Amazon Privacy Policy
- Amazon Return Policy
- Amazon A-to-Z Guarantee Policy

## How to Run

Install dependencies

pip install -r requirements.txt

Run ingestion

python app/ingest.py

Start backend

uvicorn app.main:app --reload

Start frontend

streamlit run ui.py

## Sample Questions
- What is Amazon return policy?
- What is Amazon A-to-Z Guarantee?
- What rules must Amazon sellers follow?
- How does Amazon handle customer privacy?