"""
construir_indice.py - Crea (o reconstruye) la base vectorial Chroma con todo el corpus.

Corpus:
  * Documentos generados desde data/xAPI-Edu-Data.csv (descripción, diccionario, estadísticas, hallazgos).
  * Resultados del modelo predictivo (data/metricas_modelo_mlp.json).
  * PDF y TXT de la carpeta docs/ (reglamento estudiantil u otros).

Uso:  python construir_indice.py            (reconstruye desde cero)
      python construir_indice.py --agregar  (agrega sin borrar lo existente)
"""
import sys
import time

import corpus
import rag_core


def construir(cfg: rag_core.Config | None = None, reiniciar: bool = True, vs=None) -> dict:
    cfg = cfg or rag_core.Config()
    vs = vs or rag_core.conectar_vectorstore(cfg)
    if reiniciar:
        vs.reset_collection()
    docs = (corpus.documentos_desde_csv(incluir_fichas=cfg.incluir_fichas) + corpus.documento_modelo()
            + corpus.documentos_externos())
    t0 = time.perf_counter()
    n = rag_core.indexar(vs, docs, cfg)
    info = {"documentos": len({d.metadata["doc_id"] for d in docs}), "paginas_o_secciones": len(docs),
            "fragmentos": n, "segundos": round(time.perf_counter() - t0, 1)}
    rag_core.guardar_info_indice(cfg, info)
    return info


if __name__ == "__main__":
    cfg = rag_core.Config()
    estado = rag_core.verificar_ollama(cfg)
    if not estado["ok"]:
        print("Ollama no está listo:", estado)
        print(f"Inicia Ollama y ejecuta: ollama pull {cfg.llm_model}  y  ollama pull {cfg.embedding_model}")
        sys.exit(1)
    print("Indexando con embeddings de", cfg.embedding_model, "...")
    try:
        print(construir(cfg, reiniciar="--agregar" not in sys.argv))
    except Exception as e:  # noqa: BLE001
        print("Error:", rag_core.explicar_error(e, cfg))
        sys.exit(1)
