"""
build_index.py — Run this ONCE before launching the Streamlit app.

Downloads the five legal PDFs from Google Drive, chunks them, and
builds the FAISS vector index that the app reads at runtime.

Usage:
    python build_index.py

Requirements:
    pip install langchain langchain-community langchain-text-splitters \
                faiss-cpu pypdf sentence-transformers gdown
"""

import os
import gdown
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Document registry ─────────────────────────────────────────────────────────
DOCUMENTS = {
    "UU_PDP.pdf":             "1gwzDwZZoqorirb9XXRE-acXHY9URYIrB",
    "UU_ITE.pdf":             "1DSdMaL2cJGO4__m2CbB1MKDRDnsBb56M",
    "UU_ITE_AmandemenI.pdf":  "1P9C3y6TG98CK9ObB_Ji-nIwP9c_2YfYw",
    "UU_ITE_AmandemenII.pdf": "1TLL4pw8Bk1Q3BOTn7EHOqtRd2X4nsRW4",
    "ALU_Regulations.pdf":    "1pcHYYEUYSdwYPS1Y9DrdWEbqqm0yJ1uk",
}

DOC_LABELS = {
    "UU_PDP.pdf":             "Personal Data Protection Law (UU PDP)",
    "UU_ITE.pdf":             "Electronic Information & Transactions Law (UU ITE)",
    "UU_ITE_AmandemenI.pdf":  "UU ITE Amendment I",
    "UU_ITE_AmandemenII.pdf": "UU ITE Amendment II",
    "ALU_Regulations.pdf":    "ALU University Regulations",
}


def download_documents(dest_dir: str = "documents") -> list[str]:
    os.makedirs(dest_dir, exist_ok=True)
    downloaded = []
    print("⏳ Downloading legal documents…\n")

    for filename, file_id in DOCUMENTS.items():
        path = os.path.join(dest_dir, filename)
        if os.path.exists(path):
            print(f"⏭️  {filename} already exists, skipping.")
            downloaded.append(path)
            continue
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}", path, quiet=False
            )
            size_kb = os.path.getsize(path) / 1024
            print(f"✅ {filename} downloaded ({size_kb:.1f} KB)")
            downloaded.append(path)
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")

    print(f"\n📂 Downloaded {len(downloaded)} of {len(DOCUMENTS)} documents.")
    return downloaded


def load_and_chunk(doc_dir: str = "documents") -> list:
    print("\n📖 Loading PDFs…\n")
    all_docs = []

    for filename, label in DOC_LABELS.items():
        path = os.path.join(doc_dir, filename)
        if not os.path.exists(path):
            print(f"⚠️  {filename} not found, skipping.")
            continue
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["source_name"] = label
            page.metadata["source_file"] = filename
        all_docs.extend(pages)
        print(f"✅ {label}: {len(pages)} pages loaded")

    print(f"\n📊 Total pages: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "Article ", "Section ", ". ", " "],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"✂️  Total chunks: {len(chunks)}")
    return chunks


def build_index(chunks: list, index_dir: str = "faiss_index") -> None:
    print("\n🔨 Building FAISS vector index…")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(index_dir)
    print(f"✅ Index saved to '{index_dir}/' ({len(chunks)} chunks indexed)")


if __name__ == "__main__":
    download_documents()
    chunks = load_and_chunk()
    build_index(chunks)
    print("\n🎉 Done! You can now run:  streamlit run app.py")
