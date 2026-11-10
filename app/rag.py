from time import perf_counter
import time 
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

LLM_MODEL = "qwen3.5:4b"
EMBED_MODEL = "qwen3-embedding:0.6b"

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    keep_alive=1800,
)

llm = OllamaLLM(
    model=LLM_MODEL,
    reasoning=False,     
    num_predict=256,      
    num_ctx=4096,        
    temperature=0.1,
    repeat_penalty=1.15,
    keep_alive="30m",     
)

db = Chroma(
    persist_directory="chroma",
    embedding_function=embeddings
)
PROMPT = """
Use apenas o contexto abaixo para responder à pergunta.
Se a resposta não estiver no contexto, diga que não encontrou a informação.
Responda em português, de forma direta, em no máximo 3 frases.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

def warmup():
    """Carrega os dois modelos na memória antes da primeira requisição."""
    embeddings.embed_query("warmup")
    llm.invoke("Responda apenas: ok")
 

def ask_rag(question: str):
    tempoInicio = perf_counter()

    docs = db.similarity_search(question, k=4)
    tempoDeBusca = perf_counter()
    
    if not docs:
        return {"Nenhum documento encontrado no ChromaDB,"
        "sources": []}

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT.format(context=context, question=question)
 
    resposta = llm.invoke(prompt)
    tempoFim = perf_counter()

    print(f"Tempo de Busca: {tempoDeBusca - tempoInicio:.2f}s | Tempo de Geração: {tempoFim - tempoInicio:.2f}s" )
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
        "sources": [
            {
                "source": doc.metadata.get("source", "fonte desconhecida"),
                "content": doc.page_content,
            }
            for doc in docs
        ],
    }