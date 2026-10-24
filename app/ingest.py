from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

DATA_PATH = "data/docs"
CHROMA_PATH = "chroma"

def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith((".txt", ".md")):
            loader = TextLoader(path, encoding="utf-8")
        else: 
            continue
        documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def ingest():
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    db.reset_collection()
    db.add_documents(chunks)

    print(f"Ingestão concluída: {len(chunks)} chunks criados.")


if __name__ == "__main__":
    ingest()
