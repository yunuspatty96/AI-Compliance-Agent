# AI Compliance Agent

A RAG-based data protection and privacy risk analyst agent for universities, built with LangChain + Groq (LLaMA 3.3) + FAISS + Streamlit.

## Overview

This Streamlit app analyzes data privacy and compliance scenarios against Indonesian legal documents:
- **UU PDP** — Personal Data Protection Law
- **UU ITE** — Electronic Information & Transactions Law
- **UU ITE Amendments I & II**
- **ALU University Regulations**

The agent uses Retrieval-Augmented Generation (RAG) with a FAISS vector store to find relevant legal clauses and generate structured compliance reports or conversational responses.

## How to Run

1. Start the workflow ("Start application")
2. Open the app in your browser
3. Enter your **Groq API key** in the sidebar (get a free key at https://console.groq.com/keys)
4. Upload or ensure PDF legal documents are in the `documents/` folder
5. Build the FAISS index (first-time setup)
6. Submit a privacy scenario for analysis

## Architecture

- **Frontend**: Streamlit web app (`app.py`)
- **LLM**: LLaMA 3.3 70B via Groq API
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
- **Vector Store**: FAISS (local index saved in `faiss_index/`)
- **Document Loading**: PyPDF + LangChain

## User Preferences

- Port: 5000
- Host: 0.0.0.0
