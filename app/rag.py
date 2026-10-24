from time import perf_counter
import time 
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

llm = OllamaLLM(
    model="qwen3.5:4b",
    reasoning=False,
    num_predict=100,
    temperature=0.1,
    repeat_penalty=1.15,
    keep_alive="10m",
)

def ask_rag(question: str):
    inicio = perf_counter()

    print("1. Inicializando embeddings...")
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b"
    )

    print("2. Abrindo banco vetorial...")
    db = Chroma(
        persist_directory="chroma",
        embedding_function=embeddings
    )

    print("3. Buscando documentos...")
    docs = db.similarity_search(question, k=4)
    print(f"Busca: {time.perf_counter() - inicio:.2f}s")
    print(f"Documentos encontrados: {len(docs)}")
    

    if not docs:
        return {"Nenhum documento encontrado no ChromaDB,"
        "sources": []}

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Use apenas o contexto abaixo para responder à pergunta.
Se a resposta não estiver no contexto, diga que não encontrou a informação.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

    print("4. Chamando o modelo de linguagem...")
    llm = OllamaLLM(model="qwen3.5:4b")

    resposta = llm.invoke(prompt)

    print(f"Geração: {time.perf_counter() - inicio:.2f}s")
    print(f"Resposta: {resposta!r}")
    sources = []

    for doc in docs:
        sources.append({
                "source": doc.metadata.get(
                    "source", "fonte desconhecida"
                ),
                "content": doc.page_content
            }
        )

    return {
        "answer": resposta.strip(),
        "sources": sources
    }