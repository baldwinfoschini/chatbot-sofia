"""
evaluacion.py - PASO 5: validación integral del chatbot RAG.

Métricas por caso de prueba (pruebas/casos_prueba.csv):
  * acierto_recuperacion: el documento (y artículo) esperado aparece entre los top-k fragmentos.
  * cobertura: fracción de datos clave esperados que aparecen en la respuesta.
  * cita: la respuesta incluye al menos una cita [n].
  * abstencion: la respuesta reconoce que no tiene la información (esperado en preguntas fuera de contexto).
  * tiempos de recuperación, generación y total.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

import rag_core

FRASES_ABSTENCION = ["no encuentro", "no tengo informacion", "no dispongo", "solo puedo ayudar",
                     "no esta en los documentos", "no aparece en los documentos", "no puedo ayudarte con"]


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFD", str(t).lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def cargar_casos(ruta="pruebas/casos_prueba.csv") -> pd.DataFrame:
    df = pd.read_csv(rag_core.BASE / ruta, dtype=str).fillna("")
    df["articulo_esperado"] = df["articulo_esperado"].replace("", "0").astype(int)
    return df


def _acierto(fragmentos, fuente: str, articulo: int) -> bool:
    if not fuente:
        return True
    return any(d.metadata.get("doc_id") == fuente and (articulo == 0 or d.metadata.get("articulo") == articulo)
               for d, _ in fragmentos)


def evaluar_recuperacion(vs, casos: pd.DataFrame, k: int) -> pd.DataFrame:
    filas = []
    for c in casos.itertuples():
        frag = rag_core.recuperar(vs, c.pregunta, k)
        filas.append({"id": c.id, "categoria": c.categoria, "k": k,
                      "acierto_recuperacion": _acierto(frag, c.fuente_esperada, c.articulo_esperado),
                      "relevancia_max": max((s for _, s in frag), default=0.0)})
    return pd.DataFrame(filas)


def evaluar_respuestas(vs, llm, cfg: rag_core.Config, casos: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    filas = []
    for c in casos.itertuples():
        r = rag_core.responder(c.pregunta, vs, llm, cfg)
        resp = _norm(r["respuesta"])
        claves = [k for k in c.palabras_clave.split("|") if k]
        cobertura = sum(_norm(k) in resp for k in claves) / len(claves) if claves else None
        abst = any(f in resp for f in FRASES_ABSTENCION)
        cita = bool(rag_core.citas_usadas(r["respuesta"]))
        if c.categoria == "fuera_de_contexto":
            correcto = abst
        else:
            correcto = (cobertura or 0) >= 0.5 and cita and not abst
        filas.append({"id": c.id, "categoria": c.categoria, "pregunta": c.pregunta, "respuesta": r["respuesta"],
                      "fuentes": " | ".join(rag_core.etiqueta_fuente(d) for d, _ in r["fuentes"]),
                      "acierto_recuperacion": _acierto(r["fuentes"], c.fuente_esperada, c.articulo_esperado),
                      "cobertura": cobertura, "cita": cita, "abstencion": abst, "correcto": correcto,
                      "relevancia_max": r["relevancia_max"], "t_recuperacion_s": r["t_recuperacion"],
                      "t_generacion_s": r["t_generacion"], "t_total_s": r["t_total"]})
        if verbose:
            print(f"{c.id} {'OK ' if correcto else 'X  '} {r['t_total']:.1f}s  {c.pregunta[:70]}")
    return pd.DataFrame(filas)


def resumen(res: pd.DataFrame) -> pd.DataFrame:
    return res.groupby("categoria").agg(
        casos=("id", "count"), acierto_recuperacion=("acierto_recuperacion", "mean"),
        cobertura=("cobertura", "mean"), con_cita=("cita", "mean"), abstencion=("abstencion", "mean"),
        correctos=("correcto", "mean"), t_medio_s=("t_total_s", "mean"), t_max_s=("t_total_s", "max")).round(3)
