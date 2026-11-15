"""
corpus.py - Construcción del corpus documental del chatbot.

Convierte el dataset tabular xAPI-Edu-Data en documentos de texto en español
(descripción, diccionario de variables, estadísticas por grupo, factores de riesgo
y hallazgos del modelo predictivo) y carga los PDF/TXT que se pongan en la carpeta docs/
(por ejemplo, el reglamento estudiantil). Cada documento lleva metadatos para citar la fuente.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "data" / "xAPI-Edu-Data.csv"
METRICAS_PATH = BASE / "data" / "metricas_modelo_mlp.json"
DOCS_DIR = BASE / "docs"

FUENTE_DATASET = "Aljarah, I. (2016). xAPI-Edu-Data (Students' Academic Performance). Kaggle."

NIVEL = {"L": "bajo (L, 0-69 puntos)", "M": "medio (M, 70-89 puntos)", "H": "alto (H, 90-100 puntos)"}
NIVEL_CORTO = {"L": "bajo", "M": "medio", "H": "alto"}
PAISES = {"KW": "Kuwait", "KuwaIT": "Kuwait", "lebanon": "Líbano", "Lebanon": "Líbano", "Egypt": "Egipto",
          "SaudiArabia": "Arabia Saudita", "USA": "Estados Unidos", "Jordan": "Jordania",
          "venzuela": "Venezuela", "Iran": "Irán", "Tunis": "Túnez", "Morocco": "Marruecos",
          "Syria": "Siria", "Palestine": "Palestina", "Iraq": "Irak", "Lybia": "Libia"}

# Etiquetas legibles de las variables categóricas
VARIABLES_CAT = {
    "gender": ("género", {"M": "masculino", "F": "femenino"}),
    "NationalITy": ("nacionalidad", PAISES),
    "StageID": ("nivel educativo", {"lowerlevel": "primaria", "MiddleSchool": "secundaria",
                                    "HighSchool": "bachillerato"}),
    "GradeID": ("grado escolar", {}),
    "SectionID": ("sección del aula", {}),
    "Topic": ("asignatura", {"IT": "informática", "Math": "matemáticas", "Arabic": "árabe",
                             "Science": "ciencias", "English": "inglés", "Quran": "Corán",
                             "Spanish": "español", "French": "francés", "History": "historia",
                             "Biology": "biología", "Chemistry": "química", "Geology": "geología"}),
    "Semester": ("semestre", {"F": "primer semestre", "S": "segundo semestre"}),
    "Relation": ("padre o madre responsable", {"Father": "padre", "Mum": "madre"}),
    "ParentAnsweringSurvey": ("respuesta de los padres a la encuesta escolar",
                              {"Yes": "sí respondieron", "No": "no respondieron"}),
    "ParentschoolSatisfaction": ("satisfacción de los padres con la escuela",
                                 {"Good": "buena", "Bad": "mala"}),
    "StudentAbsenceDays": ("días de ausencia del estudiante",
                           {"Under-7": "menos de 7 días", "Above-7": "más de 7 días"}),
}
VARIABLES_NUM = {
    "raisedhands": "veces que levantó la mano en clase",
    "VisITedResources": "recursos del curso visitados",
    "AnnouncementsView": "anuncios consultados",
    "Discussion": "participaciones en foros de discusión",
}


def _pct(x: float) -> str:
    return f"{x * 100:.1f} %".replace(".", ",")


def _dec(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def _num(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def _doc(texto: str, titulo: str, tipo: str, doc_id: str, **extra) -> Document:
    meta = {"fuente": titulo, "tipo": tipo, "doc_id": doc_id, "origen": FUENTE_DATASET, "pagina": 0}
    meta.update(extra)
    return Document(page_content=texto.strip(), metadata=meta)


def documentos_desde_csv(ruta_csv: Path = CSV_PATH, incluir_fichas: bool = False) -> list[Document]:
    df = pd.read_csv(ruta_csv)
    n = len(df)
    docs: list[Document] = []
    dist = df["Class"].value_counts()

    # 1. Descripción general
    docs.append(_doc(f"""
Descripción del dataset xAPI-Edu-Data (rendimiento académico de estudiantes).
El conjunto de datos fue recolectado en la plataforma de aprendizaje virtual Kalboard 360 mediante la
especificación xAPI (Experience API), que registra las acciones de los estudiantes en línea.
Contiene {n} registros de estudiantes y 16 variables, además de la variable objetivo "Class" (nivel de rendimiento).
Las variables se agrupan en cuatro tipos: demográficas (género, nacionalidad, lugar de nacimiento),
académicas (nivel educativo, grado, sección, asignatura, semestre), de comportamiento en la plataforma
(manos levantadas, recursos visitados, anuncios consultados, participación en discusiones, días de ausencia)
y de participación de los padres (quién es el responsable, si respondió la encuesta y su satisfacción con la escuela).
El nivel de rendimiento tiene tres categorías según la calificación final: bajo (L) de 0 a 69 puntos,
medio (M) de 70 a 89 puntos y alto (H) de 90 a 100 puntos.
Distribución: {dist['M']} estudiantes en nivel medio ({_pct(dist['M']/n)}), {dist['H']} en nivel alto ({_pct(dist['H']/n)})
y {dist['L']} en nivel bajo ({_pct(dist['L']/n)}).
La mayoría de los estudiantes son de Kuwait ({(df.NationalITy=='KW').sum()}) y Jordania ({(df.NationalITy=='Jordan').sum()}).
Referencia: {FUENTE_DATASET}
""", "Descripción del dataset xAPI-Edu-Data", "descripcion", "descripcion"))

    # 2. Diccionario de variables
    lineas = ["Diccionario de variables del dataset xAPI-Edu-Data (nombre en el archivo: significado y valores)."]
    for col, (nombre, mapa) in VARIABLES_CAT.items():
        valores = ", ".join(f"{v} = {mapa.get(v, v)}" for v in sorted(df[col].unique())[:12])
        lineas.append(f"- {col}: {nombre}. Valores: {valores}.")
    lineas.append("- PlaceofBirth: país de nacimiento del estudiante.")
    for col, nombre in VARIABLES_NUM.items():
        lineas.append(f"- {col}: número de {nombre}, de 0 a 100 (mínimo {df[col].min()}, máximo {df[col].max()}).")
    lineas.append("- Class: nivel de rendimiento. L = bajo (0-69), M = medio (70-89), H = alto (90-100).")
    docs.append(_doc("\n".join(lineas), "Diccionario de variables", "diccionario", "diccionario"))

    # 3. Indicadores de participación por nivel
    lineas = ["Indicadores de participación en la plataforma según el nivel de rendimiento (promedio, mediana y rango)."]
    for col, nombre in VARIABLES_NUM.items():
        partes = []
        for c in ["H", "M", "L"]:
            s = df.loc[df.Class == c, col]
            partes.append(f"nivel {NIVEL_CORTO[c]}: promedio {_num(s.mean())}, mediana {_num(s.median())}, "
                          f"rango {s.min()}-{s.max()}")
        lineas.append(f"- {nombre.capitalize()} ({col}): " + "; ".join(partes) + ".")
    corr = df[list(VARIABLES_NUM)].corrwith(df.Class.map({"L": 0, "M": 1, "H": 2}))
    lineas.append("Correlación con el nivel de rendimiento (0 = ninguna, 1 = total): " + ", ".join(
        f"{VARIABLES_NUM[c]} {corr[c]:.2f}".replace(".", ",") for c in corr.sort_values(ascending=False).index) + ".")
    lineas.append("Conclusión: los estudiantes de nivel alto participan mucho más que los de nivel bajo; "
                  "los recursos visitados y las manos levantadas son los indicadores más relacionados con el rendimiento.")
    docs.append(_doc("\n".join(lineas), "Participación en la plataforma por nivel de rendimiento",
                     "estadistica", "participacion_por_nivel"))

    # 4. Un documento por variable categórica
    for col, (nombre, mapa) in VARIABLES_CAT.items():
        if col == "GradeID":
            orden = sorted(df[col].unique())
        else:
            orden = df[col].value_counts().index
        lineas = [f"Rendimiento académico según {nombre} ({col})."]
        for v in orden:
            sub = df[df[col] == v]
            pc = sub.Class.value_counts(normalize=True)
            lineas.append(
                f"- {mapa.get(v, v).capitalize()} ({v}): {len(sub)} estudiantes. Nivel alto {_pct(pc.get('H', 0))}, "
                f"medio {_pct(pc.get('M', 0))}, bajo {_pct(pc.get('L', 0))}. Promedio de manos levantadas "
                f"{_num(sub.raisedhands.mean())} y de recursos visitados {_num(sub.VisITedResources.mean())}.")
        tasa_l = df.groupby(col).Class.apply(lambda s: (s == "L").mean())
        tasa_l = tasa_l[df[col].value_counts() >= 10]
        if len(tasa_l) > 1:
            peor, mejor = tasa_l.idxmax(), tasa_l.idxmin()
            lineas.append(f"El mayor porcentaje de bajo rendimiento se da en {mapa.get(peor, peor)} ({_pct(tasa_l[peor])}) "
                          f"y el menor en {mapa.get(mejor, mejor)} ({_pct(tasa_l[mejor])}) (solo grupos con 10 o más estudiantes).")
        docs.append(_doc("\n".join(lineas), f"Rendimiento según {nombre}", "estadistica", f"por_{col}", variable=col))

    # 5. Factores de riesgo
    def tasa(mask):
        s = df[mask]
        return len(s), (s.Class == "L").mean()
    base_l = (df.Class == "L").mean()
    n1, t1 = tasa(df.StudentAbsenceDays == "Above-7")
    n2, t2 = tasa(df.StudentAbsenceDays == "Under-7")
    n3, t3 = tasa(df.ParentAnsweringSurvey == "No")
    n4, t4 = tasa(df.ParentschoolSatisfaction == "Bad")
    q = df.VisITedResources.quantile(0.25)
    n5, t5 = tasa(df.VisITedResources <= q)
    n6, t6 = tasa((df.StudentAbsenceDays == "Above-7") & (df.ParentAnsweringSurvey == "No"))
    n7, t7 = tasa((df.StudentAbsenceDays == "Under-7") & (df.raisedhands >= 50))
    docs.append(_doc(f"""
Factores de riesgo de bajo rendimiento académico identificados en los datos.
En todo el dataset, el {_pct(base_l)} de los estudiantes tiene rendimiento bajo.
- Ausencias: con más de 7 días de ausencia, el {_pct(t1)} de los estudiantes ({n1} casos) tiene rendimiento bajo;
  con menos de 7 días, solo el {_pct(t2)} ({n2} casos). Es el factor más fuerte.
- Participación de los padres: cuando los padres no respondieron la encuesta escolar, el {_pct(t3)} tiene rendimiento bajo ({n3} casos).
  Cuando la satisfacción de los padres con la escuela es mala, el {_pct(t4)} tiene rendimiento bajo ({n4} casos).
- Uso de la plataforma: entre el 25 % de estudiantes que menos recursos visita ({q:.0f} recursos o menos),
  el {_pct(t5)} tiene rendimiento bajo ({n5} casos).
- Combinación de riesgo: más de 7 ausencias y padres que no respondieron la encuesta: {_pct(t6)} con rendimiento bajo ({n6} casos).
- Factor protector: menos de 7 ausencias y 50 o más manos levantadas: solo {_pct(t7)} con rendimiento bajo ({n7} casos).
""", "Factores de riesgo de bajo rendimiento", "hallazgo", "factores_riesgo"))

    # 6. Perfiles típicos por nivel
    lineas = ["Perfil típico del estudiante en cada nivel de rendimiento (valores más frecuentes y promedios)."]
    for c in ["H", "M", "L"]:
        s = df[df.Class == c]
        lineas.append(
            f"- Nivel {NIVEL[c]}: {len(s)} estudiantes. Promedios: {_num(s.raisedhands.mean())} manos levantadas, "
            f"{_num(s.VisITedResources.mean())} recursos visitados, {_num(s.AnnouncementsView.mean())} anuncios, "
            f"{_num(s.Discussion.mean())} discusiones. {_pct((s.StudentAbsenceDays=='Under-7').mean())} con menos de 7 ausencias; "
            f"{_pct((s.ParentAnsweringSurvey=='Yes').mean())} con padres que respondieron la encuesta; "
            f"{_pct((s.Relation=='Mum').mean())} con la madre como responsable; {_pct((s.gender=='F').mean())} mujeres.")
    docs.append(_doc("\n".join(lineas), "Perfiles típicos por nivel de rendimiento", "hallazgo", "perfiles"))

    # 7. Fichas individuales (opcional)
    if incluir_fichas:
        for i, r in df.iterrows():
            docs.append(_doc(
                f"Ficha del estudiante {i + 1}: {VARIABLES_CAT['gender'][1][r.gender]}, nacionalidad "
                f"{PAISES.get(r.NationalITy, r.NationalITy)}, {VARIABLES_CAT['StageID'][1][r.StageID]} grado {r.GradeID}, "
                f"asignatura {VARIABLES_CAT['Topic'][1].get(r.Topic, r.Topic)}, {VARIABLES_CAT['Semester'][1][r.Semester]}. "
                f"Levantó la mano {r.raisedhands} veces, visitó {r.VisITedResources} recursos, vio {r.AnnouncementsView} "
                f"anuncios y participó {r.Discussion} veces en discusiones. Ausencias: "
                f"{VARIABLES_CAT['StudentAbsenceDays'][1][r.StudentAbsenceDays]}. Rendimiento: {NIVEL[r.Class]}.",
                f"Ficha del estudiante {i + 1}", "ficha", f"ficha_{i + 1}", clase=r.Class))
    return docs


def documento_modelo(ruta: Path = METRICAS_PATH) -> list[Document]:
    """Resultados del modelo predictivo MLP desarrollado en el ACA 2 de Aprendizaje Automático."""
    if not ruta.exists():
        return []
    m = json.loads(ruta.read_text(encoding="utf-8"))
    fila = next(r for r in m["metricas_test"] if r["Modelo"] == "MLP final")
    imp = ", ".join(f"{k} ({v:.3f})".replace(".", ",") for k, v in list(m["importancia"].items())[:6])
    rep = m["reporte"]
    texto = f"""
Hallazgos del modelo predictivo de rendimiento (red neuronal MLP, proyecto de Aprendizaje Automático).
Se entrenó un Perceptrón Multicapa con capas ocultas de {', '.join(map(str, m['capas']))} neuronas para predecir el nivel
de rendimiento (bajo, medio, alto) a partir de las variables del dataset xAPI-Edu-Data.
Resultados en el conjunto de prueba ({m['particion']['test']} estudiantes): exactitud {_pct(fila['Accuracy'])},
F1 macro {_dec(fila['F1 macro'])}; en validación cruzada de 5 particiones, F1 macro {_dec(fila['F1 macro CV (5 part.)'])}.
El modelo detecta al {_pct(rep['L']['recall'])} de los estudiantes de rendimiento bajo (recall de la clase L).
Su error más frecuente es clasificar como nivel medio a estudiantes de nivel alto.
Variables más importantes (caída del F1 al permutarlas): {imp}.
Conclusión: las ausencias, la participación en clase y el involucramiento de la familia son las señales más útiles
para anticipar el rendimiento y activar alertas tempranas.
"""
    return [_doc(texto, "Hallazgos del modelo predictivo MLP", "modelo", "modelo_mlp",
                 origen="Proyecto ACA 2 Aprendizaje Automático (CUN)")]


def _limpiar_encabezados(paginas: list[Document]) -> None:
    """Quita encabezados/pies repetidos y números de página para que no contaminen los fragmentos."""
    import re
    from collections import Counter

    if len(paginas) < 2:
        return
    conteo = Counter(l.strip() for p in paginas for l in set(p.page_content.splitlines()) if l.strip())
    repetidas = {l for l, c in conteo.items() if c >= max(2, len(paginas) * 0.6)}
    for p in paginas:
        lineas = [l for l in p.page_content.splitlines()
                  if l.strip() not in repetidas and not re.fullmatch(r"\s*P[áa]gina \d+\s*", l)]
        p.page_content = "\n".join(lineas)


def _por_articulos(paginas: list[Document], ruta: Path) -> list[Document]:
    """Si el texto está organizado en 'Artículo N.', crea un documento por artículo (con su título y página)."""
    import re

    texto, inicios = "", []
    for p in paginas:
        inicios.append((len(texto), int(p.metadata.get("page", 0)) + 1))
        texto += p.page_content + "\n"
    marcas = list(re.finditer(r"(?m)^Artículo (\d+)\.", texto))
    if len(marcas) < 3:
        return []
    pagina_de = lambda pos: max(pg for ini, pg in inicios if ini <= pos)
    titulos = [(m.start(), m.group(0).strip()) for m in re.finditer(r"(?m)^(TÍTULO [^\n]+|DISPOSICIONES [^\n]+)$", texto)]
    docs = []
    for i, m in enumerate(marcas):
        fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(texto)
        cuerpo = re.sub(r"(?m)^(TÍTULO [^\n]+|DISPOSICIONES [^\n]+)$", "", texto[m.start():fin]).strip()
        titulo = next((t for pos, t in reversed(titulos) if pos < m.start()), "")
        docs.append(Document(page_content=(f"{titulo}\n" if titulo else "") + " ".join(cuerpo.split()),
                             metadata={"fuente": ruta.stem.replace("_", " "), "tipo": "documento",
                                       "doc_id": f"doc_{ruta.stem}", "origen": ruta.name,
                                       "pagina": pagina_de(m.start()), "articulo": int(m.group(1)),
                                       "titulo": titulo}))
    return docs


def documentos_externos(carpeta: Path = DOCS_DIR, rutas: list[Path] | None = None) -> list[Document]:
    """Carga PDF y TXT de la carpeta docs/ (reglamentos, manuales, guías)."""
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    docs: list[Document] = []
    for ruta in rutas or sorted(carpeta.glob("*")):
        ruta = Path(ruta)
        if ruta.suffix.lower() == ".pdf":
            paginas = PyPDFLoader(str(ruta)).load()
        elif ruta.suffix.lower() in {".txt", ".md"}:
            paginas = TextLoader(str(ruta), encoding="utf-8").load()
        else:
            continue
        _limpiar_encabezados(paginas)
        articulos = _por_articulos(paginas, ruta)
        if articulos:
            docs.extend(articulos)
            continue
        for p in paginas:
            pag = int(p.metadata.get("page", 0)) + 1 if ruta.suffix.lower() == ".pdf" else 0
            p.metadata = {"fuente": ruta.stem.replace("_", " "), "tipo": "documento", "doc_id": f"doc_{ruta.stem}",
                          "origen": ruta.name, "pagina": pag}
        docs.extend(paginas)
    return docs
