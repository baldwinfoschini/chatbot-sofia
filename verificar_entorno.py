"""
verificar_entorno.py - Diagnóstico rápido antes de ejecutar la app.

Revisa: versión de Python, librerías, archivo .env, servidor Ollama, modelos descargados y base vectorial.
Uso:  python verificar_entorno.py
"""
import importlib
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OK, FALLA, AVISO = "[OK]   ", "[FALLA]", "[AVISO]"
problemas = 0


def linea(estado: str, texto: str, ayuda: str = "") -> None:
    global problemas
    if estado == FALLA:
        problemas += 1
    print(f"{estado} {texto}" + (f"\n         -> {ayuda}" if ayuda else ""))


print("Verificación del entorno · Chatbot RAG Sofía\n")

# 1. Python
v = sys.version_info
linea(OK if v >= (3, 10) else FALLA, f"Python {v.major}.{v.minor}.{v.micro}",
      "" if v >= (3, 10) else "Se necesita Python 3.10 o superior.")

# 2. Librerías
faltan = []
for modulo in ["streamlit", "langchain_core", "langchain_community", "langchain_ollama", "langchain_chroma",
               "langchain_text_splitters", "chromadb", "pypdf", "dotenv", "pandas", "requests", "altair"]:
    try:
        importlib.import_module(modulo)
    except ImportError:
        faltan.append(modulo)
linea(OK if not faltan else FALLA, "Librerías de requirements.txt" + (f" (faltan: {', '.join(faltan)})" if faltan else ""),
      "Ejecuta: pip install -r requirements.txt" if faltan else "")
if faltan:
    print(f"\n{problemas} problema(s) encontrado(s). Corrige lo anterior y vuelve a ejecutar.")
    sys.exit(1)

# 3. Archivo .env
if (BASE / ".env").exists():
    linea(OK, "Archivo .env encontrado")
else:
    linea(AVISO, "No existe .env: se usan los valores por defecto",
          "Copia .env.example como .env (Windows: copy .env.example .env)")

import rag_core  # noqa: E402  (después de comprobar las librerías)

cfg = rag_core.Config()
print(f"         LLM: {cfg.llm_model} · embeddings: {cfg.embedding_model} · servidor: {cfg.ollama_url}")

# 4. Ollama y modelos
estado = rag_core.verificar_ollama(cfg)
if not estado["servidor"]:
    linea(FALLA, f"Ollama no responde en {cfg.ollama_url}",
          "Abre la aplicación de Ollama o ejecuta: ollama serve")
else:
    linea(OK, f"Ollama responde ({len(estado['modelos'])} modelos instalados)")
    for m in sorted({cfg.llm_model, cfg.embedding_model}):
        linea(OK if m not in estado["falta"] else FALLA, f"Modelo {m}",
              f"Ejecuta: ollama pull {m}" if m in estado["falta"] else "")

# 5. Base vectorial
try:
    from langchain_core.embeddings import Embeddings

    class _SinEmbeddings(Embeddings):  # solo para contar, sin llamar a Ollama
        def embed_documents(self, textos):
            raise RuntimeError

        def embed_query(self, texto):
            raise RuntimeError

    n = rag_core.conectar_vectorstore(cfg, _SinEmbeddings())._collection.count()
    info = rag_core.info_indice(cfg)
    if n == 0:
        linea(AVISO, "La base vectorial está vacía", "Ejecuta: python construir_indice.py")
    else:
        linea(OK, f"Base vectorial con {n} fragmentos" + (f" (indexada el {info['fecha']})" if info else ""))
        if info and info.get("embedding_model") != cfg.embedding_model:
            linea(FALLA, f"El índice usa '{info['embedding_model']}' y el .env dice '{cfg.embedding_model}'",
                  "Reconstruye el índice: python construir_indice.py")
except Exception as e:  # noqa: BLE001
    linea(FALLA, f"No se pudo abrir la base vectorial: {e}", "Borra la carpeta chroma_db y vuelve a indexar.")

# 6. Documentos
pdfs = list((BASE / "docs").glob("*.pdf"))
linea(OK if pdfs else AVISO, f"{len(pdfs)} PDF en docs/", "" if pdfs else "Agrega el reglamento u otros documentos.")

print("\nTodo listo: ejecuta  streamlit run app.py" if problemas == 0
      else f"\n{problemas} problema(s) encontrado(s). Corrige lo anterior y vuelve a ejecutar.")
sys.exit(0 if problemas == 0 else 1)
