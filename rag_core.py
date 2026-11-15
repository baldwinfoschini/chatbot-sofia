"""
rag_core.py - Núcleo del sistema RAG (Retrieval-Augmented Generation).

PASO 1  Conexión y recuperación de la base vectorial (Chroma persistente).
PASO 2  Configuración del LLM local en Ollama (phi3) con variables de entorno (.env).
        Los embeddings usan nomic-embed-text (también en Ollama): las versiones recientes de Ollama
        solo generan embeddings con modelos especializados.
PASO 3  Canalización RAG: plantilla de prompt, inyección de contexto y citas de fuentes.

Lo usan la app de Streamlit (app.py), el script de indexación (construir_indice.py) y el notebook.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")


# ═════════════════════════════════════════════════════════════════════════════
# Configuración (todo sale de variables de entorno, con valores por defecto)
# ═════════════════════════════════════════════════════════════════════════════
def _env(nombre: str, defecto, tipo=str):
    valor = os.getenv(nombre)
    return defecto if valor in (None, "") else tipo(valor)


@dataclass
class Config:
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "phi3"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "nomic-embed-text"))
    temperature: float = field(default_factory=lambda: _env("LLM_TEMPERATURE", 0.2, float))
    top_p: float = field(default_factory=lambda: _env("LLM_TOP_P", 0.9, float))
    num_ctx: int = field(default_factory=lambda: _env("LLM_NUM_CTX", 4096, int))
    num_predict: int = field(default_factory=lambda: _env("LLM_NUM_PREDICT", 512, int))
    chroma_dir: str = field(default_factory=lambda: _env("CHROMA_DIR", str(BASE / "chroma_db")))
    collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "aca2_educacion_rag"))
    top_k: int = field(default_factory=lambda: _env("TOP_K", 4, int))
    chunk_size: int = field(default_factory=lambda: _env("CHUNK_SIZE", 700, int))
    chunk_overlap: int = field(default_factory=lambda: _env("CHUNK_OVERLAP", 100, int))
    umbral_relevancia: float = field(default_factory=lambda: _env("UMBRAL_RELEVANCIA", 0.0, float))
    max_turnos_historial: int = field(default_factory=lambda: _env("MAX_TURNOS_HISTORIAL", 3, int))
    incluir_fichas: bool = field(default_factory=lambda: _env("INCLUIR_FICHAS", "false").lower() == "true")
    log_dir: str = field(default_factory=lambda: _env("LOG_DIR", str(BASE / "logs")))

    def __post_init__(self):
        # En Windows "localhost" puede resolverse a IPv6 (::1) y Ollama solo escucha en IPv4.
        self.ollama_url = self.ollama_url.replace("localhost", "127.0.0.1").rstrip("/")


# ═════════════════════════════════════════════════════════════════════════════
# PASO 2 · Modelo de lenguaje y embeddings en Ollama
# ═════════════════════════════════════════════════════════════════════════════
def crear_llm(cfg: Config):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=cfg.llm_model, base_url=cfg.ollama_url, temperature=cfg.temperature,
                      top_p=cfg.top_p, num_ctx=cfg.num_ctx, num_predict=cfg.num_predict)


def crear_embeddings(cfg: Config):
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=cfg.embedding_model, base_url=cfg.ollama_url)


def explicar_error(e: Exception, cfg: Config) -> str:
    """Traduce los errores más comunes de Ollama a instrucciones claras."""
    texto = str(e)
    if "does not support embeddings" in texto or "--embeddings" in texto:
        return (f"El modelo '{cfg.embedding_model}' no puede generar embeddings en tu versión de Ollama. "
                "Usa un modelo de embeddings: ejecuta `ollama pull nomic-embed-text`, pon "
                "`EMBEDDING_MODEL=nomic-embed-text` en el archivo .env y reinicia la app.")
    if "not found" in texto and "model" in texto:
        return f"Falta descargar un modelo en Ollama. Ejecuta `ollama pull {cfg.llm_model}` y `ollama pull {cfg.embedding_model}`."
    if "dimension" in texto.lower():
        return ("La base vectorial se creó con otro modelo de embeddings. Pulsa 'Reconstruir índice' "
                "o borra la carpeta chroma_db y vuelve a indexar.")
    if "Connection" in texto or "refused" in texto:
        return "No hay conexión con Ollama. Abre la aplicación de Ollama o ejecuta `ollama serve`."
    return texto


def verificar_ollama(cfg: Config) -> dict:
    """Comprueba que el servidor Ollama responde y que los modelos configurados están descargados."""
    import requests

    error = ""
    for _ in range(2):  # un reintento: si phi3 se está cargando, el equipo puede tardar en responder
        try:
            r = requests.get(f"{cfg.ollama_url}/api/tags", timeout=10)
            r.raise_for_status()
            modelos = [m["name"] for m in r.json().get("models", [])]
            tiene = lambda nombre: any(m == nombre or m.startswith(nombre + ":") for m in modelos)
            return {"ok": tiene(cfg.llm_model) and tiene(cfg.embedding_model), "servidor": True, "modelos": modelos,
                    "falta": sorted(m for m in {cfg.llm_model, cfg.embedding_model} if not tiene(m))}
        except Exception as e:  # noqa: BLE001
            error = str(e)
    return {"ok": False, "servidor": False, "modelos": [], "falta": [], "error": error}


# ═════════════════════════════════════════════════════════════════════════════
# PASO 1 · Base vectorial Chroma (persistente en disco)
# ═════════════════════════════════════════════════════════════════════════════
def _ruta(p: str) -> Path:
    ruta = Path(p)
    return ruta if ruta.is_absolute() else BASE / ruta


def conectar_vectorstore(cfg: Config, embeddings=None):
    from langchain_chroma import Chroma

    ruta = _ruta(cfg.chroma_dir)
    return Chroma(collection_name=cfg.collection, embedding_function=embeddings or crear_embeddings(cfg),
                  persist_directory=str(ruta), collection_metadata={"hnsw:space": "cosine"})


def dividir(docs: list[Document], cfg: Config) -> list[Document]:
    """Divide en fragmentos respetando artículos y párrafos; cada fragmento recibe un id estable."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
        separators=["\nArtículo ", "\nTÍTULO", "\n- ", "\n\n", "\n", ". ", " "])
    fragmentos = splitter.split_documents(docs)
    contador: dict[str, int] = {}
    for f in fragmentos:
        base = f"{f.metadata['doc_id']}-p{f.metadata.get('pagina', 0)}"
        contador[base] = contador.get(base, 0) + 1
        f.metadata["chunk_id"] = f"{base}-{contador[base]}"
        if not f.metadata.get("articulo"):
            m = re.search(r"Artículo (\d+)", f.page_content)
            f.metadata["articulo"] = int(m.group(1)) if m else 0
        f.metadata.setdefault("titulo", "")
    return fragmentos


def indexar(vs, docs: list[Document], cfg: Config) -> int:
    fragmentos = dividir(docs, cfg)
    for i in range(0, len(fragmentos), 64):
        lote = fragmentos[i:i + 64]
        vs.add_documents(lote, ids=[f.metadata["chunk_id"] for f in lote])
    return len(fragmentos)


def eliminar_documento(vs, doc_id: str) -> int:
    ids = vs.get(where={"doc_id": doc_id})["ids"]
    if ids:
        vs.delete(ids=ids)
    return len(ids)


def listar_fragmentos(vs) -> pd.DataFrame:
    datos = vs.get(include=["metadatas", "documents"])
    if not datos["ids"]:
        return pd.DataFrame(columns=["id", "fuente", "tipo", "doc_id", "pagina", "articulo", "texto"])
    df = pd.DataFrame(datos["metadatas"])
    df.insert(0, "id", datos["ids"])
    df["texto"] = datos["documents"]
    return df


def resumen_documentos(vs) -> pd.DataFrame:
    df = listar_fragmentos(vs)
    if df.empty:
        return df
    return (df.groupby(["doc_id", "fuente", "tipo"], as_index=False)
              .agg(fragmentos=("id", "count"), paginas=("pagina", "max"), caracteres=("texto", lambda s: s.str.len().sum()))
              .sort_values(["tipo", "fuente"]))


def recuperar(vs, consulta: str, k: int, filtro: dict | None = None) -> list[tuple[Document, float]]:
    """Top-k fragmentos con su relevancia (1 = idéntico, 0 = nada que ver; distancia coseno)."""
    resultados = vs.similarity_search_with_score(consulta, k=k, filter=filtro or None)
    return [(doc, max(0.0, 1.0 - float(dist))) for doc, dist in resultados]


# ═════════════════════════════════════════════════════════════════════════════
# PASO 3 · Prompt engineering y canalización RAG
# ═════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Eres Sofía, asistente de orientación estudiantil de la Institución Educativa Virtual Horizonte.
Ayudas a estudiantes y familias con el reglamento estudiantil (permanencia, evaluación, ausencias, homologación,
convivencia, bienestar) y con los datos de rendimiento académico del estudio xAPI-Edu-Data.

Reglas obligatorias:
1. Responde SOLO con la información del CONTEXTO. No uses conocimiento externo ni inventes cifras, plazos o artículos.
2. Cita la fuente de cada dato con el número del fragmento entre corchetes, por ejemplo [1] o [2].
   Si el dato viene del reglamento, menciona también el artículo.
3. Si el contexto no contiene la respuesta, di exactamente: "No encuentro esa información en los documentos disponibles."
   y sugiere a quién acudir o qué preguntar.
4. Si la pregunta no tiene relación con la vida académica o los datos del estudio, responde con amabilidad que solo
   puedes ayudar con esos temas.
5. Responde en español, de forma clara, cálida y breve (máximo 6 oraciones o una lista corta)."""

PLANTILLA_USUARIO = """CONTEXTO (fragmentos recuperados de los documentos):
{contexto}

PREGUNTA: {pregunta}

Responde usando solo el CONTEXTO y cita los fragmentos con [n]."""

RESPUESTA_SIN_CONTEXTO = ("No encuentro esa información en los documentos disponibles. Puedo ayudarte con el reglamento "
                          "estudiantil (permanencia, evaluación, ausencias, homologación, convivencia) o con los datos "
                          "de rendimiento académico del estudio xAPI-Edu-Data.")


def etiqueta_fuente(doc: Document) -> str:
    m = doc.metadata
    partes = [m.get("fuente", "Documento")]
    if m.get("articulo"):
        partes.append(f"Art. {m['articulo']}")
    if m.get("pagina"):
        partes.append(f"pág. {m['pagina']}")
    return " · ".join(partes)


def formatear_contexto(fragmentos: list[tuple[Document, float]]) -> str:
    return "\n\n".join(f"[{i}] ({etiqueta_fuente(d)})\n{d.page_content}" for i, (d, _) in enumerate(fragmentos, 1))


def construir_mensajes(pregunta: str, fragmentos, historial: list[dict], cfg: Config) -> list:
    mensajes = [SystemMessage(content=SYSTEM_PROMPT)]
    # memoria conversacional: últimos turnos (sin el contexto viejo, para no saturar la ventana de phi3)
    for msg in historial[-2 * cfg.max_turnos_historial:]:
        clase = HumanMessage if msg["role"] == "user" else AIMessage
        mensajes.append(clase(content=msg["content"]))
    mensajes.append(HumanMessage(content=PLANTILLA_USUARIO.format(contexto=formatear_contexto(fragmentos),
                                                                  pregunta=pregunta)))
    return mensajes


def responder(pregunta: str, vs, llm, cfg: Config, historial: list[dict] | None = None,
              filtro: dict | None = None, k: int | None = None) -> dict:
    """Ejecuta el ciclo RAG completo y devuelve respuesta, fuentes y tiempos."""
    historial = historial or []
    t0 = time.perf_counter()
    fragmentos = recuperar(vs, pregunta, k or cfg.top_k, filtro)
    t1 = time.perf_counter()
    mejor = max((s for _, s in fragmentos), default=0.0)
    if not fragmentos or mejor < cfg.umbral_relevancia:
        return {"respuesta": RESPUESTA_SIN_CONTEXTO, "fuentes": fragmentos, "sin_contexto": True,
                "t_recuperacion": t1 - t0, "t_generacion": 0.0, "t_total": t1 - t0, "relevancia_max": mejor}
    mensajes = construir_mensajes(pregunta, fragmentos, historial, cfg)
    respuesta = llm.invoke(mensajes).content.strip()
    t2 = time.perf_counter()
    return {"respuesta": respuesta, "fuentes": fragmentos, "sin_contexto": False, "t_recuperacion": t1 - t0,
            "t_generacion": t2 - t1, "t_total": t2 - t0, "relevancia_max": mejor}


def citas_usadas(respuesta: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", respuesta)})


# ═════════════════════════════════════════════════════════════════════════════
# Monitoreo: información del índice y registro de consultas
# ═════════════════════════════════════════════════════════════════════════════
def guardar_info_indice(cfg: Config, info: dict) -> None:
    import json
    from datetime import datetime

    ruta = _ruta(cfg.chroma_dir)
    ruta.mkdir(parents=True, exist_ok=True)
    datos = {**info, "fecha": datetime.now().isoformat(timespec="seconds"), "embedding_model": cfg.embedding_model,
             "chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap}
    (ruta / "indice_info.json").write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def info_indice(cfg: Config) -> dict:
    import json

    archivo = _ruta(cfg.chroma_dir) / "indice_info.json"
    return json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {}


def registrar_consulta(cfg: Config, pregunta: str, resultado: dict, top_k: int, filtro: dict | None) -> None:
    """Guarda cada consulta en logs/consultas.jsonl (una línea JSON por consulta) para el monitoreo."""
    import json
    from datetime import datetime

    usadas = set(citas_usadas(resultado["respuesta"]))
    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "pregunta": pregunta,
        "top_k": top_k,
        "filtro": ",".join(filtro["tipo"]["$in"]) if filtro else "",
        "t_recuperacion_s": round(resultado["t_recuperacion"], 4),
        "t_generacion_s": round(resultado["t_generacion"], 3),
        "t_total_s": round(resultado["t_total"], 3),
        "relevancia_max": round(resultado["relevancia_max"], 4),
        "n_fuentes": len(resultado["fuentes"]),
        "con_cita": bool(usadas),
        "abstencion": "no encuentro esa información" in resultado["respuesta"].lower() or resultado["sin_contexto"],
        "fuentes_citadas": [etiqueta_fuente(d) for i, (d, _) in enumerate(resultado["fuentes"], 1) if i in usadas],
        "documentos_citados": sorted({d.metadata.get("fuente", "") for i, (d, _) in enumerate(resultado["fuentes"], 1)
                                      if i in usadas}),
        "modelo": cfg.llm_model,
    }
    carpeta = _ruta(cfg.log_dir)
    carpeta.mkdir(parents=True, exist_ok=True)
    with open(carpeta / "consultas.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def leer_consultas(cfg: Config) -> pd.DataFrame:
    archivo = _ruta(cfg.log_dir) / "consultas.jsonl"
    if not archivo.exists() or archivo.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_json(archivo, lines=True)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def borrar_consultas(cfg: Config) -> None:
    archivo = _ruta(cfg.log_dir) / "consultas.jsonl"
    if archivo.exists():
        archivo.unlink()
