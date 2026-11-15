"""
app.py - PASO 4: Interfaz gráfica del chatbot RAG de orientación estudiantil.

Basado en CHAT_ACA1.py (ACA 1). Cambios principales:
  * La base vectorial Chroma queda persistente en disco y se administra desde la app.
  * Configuración del LLM (Ollama, phi3) en variables de entorno (.env).
  * Prompt con instrucciones anti-alucinación y citas numeradas de las fuentes.
  * Módulo de gestión documental: documentos, metadatos, fragmentos, prueba de recuperación,
    carga y eliminación de documentos.

Ejecutar:  streamlit run app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import construir_indice
import corpus
import rag_core

BASE = Path(__file__).resolve().parent
ASSETS = BASE / "assets"
AVATAR = {"user": str(ASSETS / "avatar_estudiante.png"), "assistant": str(ASSETS / "avatar_asistente.png")}
TIPOS = {"documento": "Reglamento y documentos", "estadistica": "Estadísticas", "hallazgo": "Hallazgos",
         "descripcion": "Descripción del dataset", "diccionario": "Diccionario", "modelo": "Modelo predictivo",
         "ficha": "Fichas de estudiantes"}
SUGERIDAS = [
    "¿Cuántas ausencias puedo tener sin perder una asignatura?",
    "¿Qué necesito para homologar una materia de otro colegio?",
    "¿Cómo se calcula la nota final de una asignatura?",
    "¿Qué factores se relacionan con el bajo rendimiento?",
]

st.set_page_config(page_title="Sofía · Orientación estudiantil", page_icon=str(ASSETS / "favicon.png"),
                   layout="wide")


# ─────────────────────────────────────────────
# 1. RECURSOS (se crean una sola vez)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def recursos():
    cfg = rag_core.Config()
    llm = rag_core.crear_llm(cfg)                 # PASO 2: phi3 en Ollama
    embeddings = rag_core.crear_embeddings(cfg)
    return cfg, llm, embeddings


@st.cache_data(ttl=20, show_spinner=False)
def estado_ollama(_cfg):
    return rag_core.verificar_ollama(_cfg)


cfg, llm, embeddings = recursos()
estado = estado_ollama(cfg)
if not estado["ok"]:
    estado_ollama.clear()  # un fallo no se guarda: en la siguiente interacción se vuelve a verificar
# PASO 1: la conexión a Chroma se abre en cada ejecución, así la app siempre ve el índice actual
# aunque se haya reconstruido desde la terminal con construir_indice.py.
vs = rag_core.conectar_vectorstore(cfg, embeddings)

# Streamlit marca la página como inglés y el navegador intenta traducirla: se declara español y "no traducir".
_JS_ES = ("const d = (window.parent || window).document.documentElement;"
          "d.lang = 'es'; d.setAttribute('translate', 'no'); d.classList.add('notranslate');")
try:  # Streamlit reciente
    st.html(f"<script>{_JS_ES}</script>", unsafe_allow_javascript=True)
except TypeError:  # versiones anteriores
    components.html(f"<script>{_JS_ES}</script>", height=0)


def n_fragmentos() -> int:
    try:
        return vs._collection.count()
    except Exception:  # noqa: BLE001
        return 0


def construir_desde_app():
    with st.spinner("Indexando el corpus con phi3… puede tardar unos minutos la primera vez."):
        try:
            info = construir_indice.construir(cfg, vs=vs)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo construir el índice. {rag_core.explicar_error(e, cfg)}")
            st.stop()
    st.session_state.aviso = (f"Base vectorial lista: {info['documentos']} documentos y {info['fragmentos']} "
                              f"fragmentos indexados en {info['segundos']} s.")
    st.rerun()


# ─────────────────────────────────────────────
# 2. BARRA LATERAL
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(AVATAR["assistant"], width=96)
    st.markdown("### Sofía\nAsistente de orientación estudiantil")
    if estado["ok"]:
        st.success(f"Ollama conectado · {cfg.llm_model} + {cfg.embedding_model}", icon=":material/check_circle:")
    elif estado["servidor"]:
        st.warning(f"Ollama responde, pero falta descargar: {', '.join(estado['falta'])}.\n\n"
                   f"Ejecuta `ollama pull {estado['falta'][0]}`", icon=":material/download:")
    else:
        st.error(f"No hay conexión con Ollama en {cfg.ollama_url}. Abre la aplicación de Ollama o ejecuta "
                 "`ollama serve` en una terminal.", icon=":material/error:")
        with st.expander("Detalle técnico"):
            st.code(estado.get("error", ""), language=None)
    if not estado["ok"] and st.button("Reintentar conexión", icon=":material/refresh:", width="stretch"):
        estado_ollama.clear()
        st.rerun()
    st.caption(f"Base vectorial: {n_fragmentos()} fragmentos indexados")

    st.divider()
    st.markdown("**Ajustes de búsqueda**")
    top_k = st.slider("Fragmentos a recuperar (top-k)", 2, 8, cfg.top_k)
    tipos_sel = st.multiselect("Buscar en", list(TIPOS), default=[], format_func=TIPOS.get,
                               placeholder="Todos los documentos")
    filtro = {"tipo": {"$in": tipos_sel}} if tipos_sel else None
    with st.expander("Parámetros del modelo"):
        st.markdown(f"- Modelo: `{cfg.llm_model}`\n- Embeddings: `{cfg.embedding_model}`\n"
                    f"- Temperatura: `{cfg.temperature}`\n- top_p: `{cfg.top_p}`\n"
                    f"- Ventana de contexto: `{cfg.num_ctx}` tokens\n- Máx. respuesta: `{cfg.num_predict}` tokens\n"
                    f"- Servidor: `{cfg.ollama_url}`")
    if st.button("Nueva conversación", icon=":material/refresh:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# 3. ENCABEZADO
# ─────────────────────────────────────────────
st.image(str(ASSETS / "banner.png"), width="stretch")
tab_chat, tab_docs, tab_mon = st.tabs([":material/forum: Chat con Sofía", ":material/folder_open: Gestión documental",
                                     ":material/monitoring: Monitoreo"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "aviso" in st.session_state:
    st.toast(st.session_state.pop("aviso"), icon=":material/check_circle:")


def mostrar_fuentes(msg: dict):
    fuentes = msg.get("fuentes") or []
    if not fuentes:
        return
    usadas = set(rag_core.citas_usadas(msg["content"]))
    with st.expander(f"Fuentes consultadas ({len(fuentes)}) · {msg['t_total']:.1f} s"):
        st.caption(f"Recuperación {msg['t_recuperacion'] * 1000:.0f} ms · Generación {msg['t_generacion']:.1f} s")
        for i, f in enumerate(fuentes, 1):
            marca = "Citada" if i in usadas else "No citada"
            st.markdown(f"**[{i}] {f['etiqueta']}** · relevancia {f['relevancia']:.2f} · {marca}")
            st.caption(f["texto"][:600] + ("…" if len(f["texto"]) > 600 else ""))


# ─────────────────────────────────────────────
# 4. MÓDULO DE CHAT (RAG con historial)
# ─────────────────────────────────────────────
with tab_chat:
    if n_fragmentos() == 0:
        st.info("La base vectorial está vacía. Constrúyela con el botón (o ejecuta `python construir_indice.py` "
                "en la terminal) y luego podrás conversar con Sofía.", icon=":material/info:")
        if st.button("Construir la base vectorial ahora", icon=":material/database:", type="primary",
                     key="construir_chat"):
            construir_desde_app()

    conversacion = st.container()  # los mensajes van aquí; la caja de texto queda siempre al final
    prompt = st.chat_input("Escribe tu pregunta aquí…", disabled=n_fragmentos() == 0)
    prompt = prompt or st.session_state.pop("pendiente", None)

    with conversacion:
        if not st.session_state.messages and not prompt:
            with st.chat_message("assistant", avatar=AVATAR["assistant"]):
                st.markdown("¡Hola! Soy **Sofía**. Puedo orientarte sobre el reglamento estudiantil (permanencia, "
                            "evaluación, ausencias, homologación, convivencia y bienestar) y sobre los datos de "
                            "rendimiento académico del estudio xAPI-Edu-Data. Siempre te digo de qué documento sale "
                            "cada respuesta.")
            st.markdown("**Prueba con una de estas preguntas:**")
            cols = st.columns(2)
            for i, q in enumerate(SUGERIDAS):
                if cols[i % 2].button(q, key=f"sug{i}", width="stretch", disabled=n_fragmentos() == 0):
                    st.session_state.pendiente = q
                    st.rerun()

        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar=AVATAR[m["role"]]):
                st.markdown(m["content"])
                if m["role"] == "assistant":
                    mostrar_fuentes(m)

        if prompt:
            with st.chat_message("user", avatar=AVATAR["user"]):
                st.markdown(prompt)
            historial = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant", avatar=AVATAR["assistant"]):
                with st.spinner("Sofía está buscando en los documentos…"):
                    try:
                        r = rag_core.responder(prompt, vs, llm, cfg, historial=historial, filtro=filtro, k=top_k)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"No pude generar la respuesta. {rag_core.explicar_error(e, cfg)}")
                        st.stop()
                msg = {"role": "assistant", "content": r["respuesta"], "t_total": r["t_total"],
                       "t_recuperacion": r["t_recuperacion"], "t_generacion": r["t_generacion"],
                       "fuentes": [{"etiqueta": rag_core.etiqueta_fuente(d), "relevancia": s,
                                    "texto": d.page_content} for d, s in r["fuentes"]]}
                st.markdown(msg["content"])
                mostrar_fuentes(msg)
            st.session_state.messages.append(msg)
            try:
                rag_core.registrar_consulta(cfg, prompt, r, top_k, filtro)
            except OSError:
                pass  # el monitoreo nunca debe impedir que el chat funcione

# ─────────────────────────────────────────────
# 5. MÓDULO DE GESTIÓN DOCUMENTAL
# ─────────────────────────────────────────────
with tab_docs:
    resumen = rag_core.resumen_documentos(vs)
    c1, c2, c3 = st.columns(3)
    c1.metric("Documentos", 0 if resumen.empty else len(resumen))
    c2.metric("Fragmentos", n_fragmentos())
    c3.metric("Páginas del reglamento", int(resumen.paginas.max()) if not resumen.empty else 0)
    st.caption(f"Colección `{cfg.collection}` · modelo de embeddings `{cfg.embedding_model}` · "
               f"carpeta `{cfg.chroma_dir}`")

    st.subheader("Documentos en la base vectorial")
    if resumen.empty:
        st.warning("No hay documentos indexados todavía.")
    else:
        vista = resumen.assign(tipo=resumen.tipo.map(lambda t: TIPOS.get(t, t)))
        st.dataframe(vista.rename(columns={"fuente": "Documento", "tipo": "Tipo", "fragmentos": "Fragmentos",
                                           "paginas": "Páginas", "caracteres": "Caracteres", "doc_id": "ID"}),
                     hide_index=True, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Explorar fragmentos y metadatos")
        if not resumen.empty:
            doc_sel = st.selectbox("Documento", resumen.doc_id, format_func=dict(zip(resumen.doc_id, resumen.fuente)).get)
            frag = rag_core.listar_fragmentos(vs)
            frag = frag[frag.doc_id == doc_sel].sort_values(["pagina", "articulo", "id"])
            st.dataframe(frag[["id", "pagina", "articulo", "texto"]], hide_index=True, width="stretch",
                         height=320, column_config={"texto": st.column_config.TextColumn("Texto", width="large")})

    with col_b:
        st.subheader("Probar la recuperación semántica")
        consulta = st.text_input("Consulta", "requisitos de permanencia")
        k_prueba = st.slider("Top-k", 1, 10, 4, key="kprueba")
        if st.button("Buscar fragmentos", icon=":material/search:", disabled=n_fragmentos() == 0):
            try:
                res = rag_core.recuperar(vs, consulta, k_prueba)
            except Exception as e:  # noqa: BLE001
                st.error(rag_core.explicar_error(e, cfg))
                st.stop()
            st.dataframe(pd.DataFrame([{"#": i, "Fuente": rag_core.etiqueta_fuente(d), "Relevancia": round(s, 3),
                                        "Texto": d.page_content[:300]} for i, (d, s) in enumerate(res, 1)]),
                         hide_index=True, width="stretch",
                         column_config={"Relevancia": st.column_config.ProgressColumn(min_value=0, max_value=1)})

    st.subheader("Administrar documentos")
    a1, a2, a3 = st.columns(3)
    with a1:
        archivo = st.file_uploader("Cargar PDF o TXT (por ejemplo, un reglamento)", type=["pdf", "txt"])
        if archivo and st.button("Indexar documento", icon=":material/upload:"):
            destino = corpus.DOCS_DIR / archivo.name
            destino.write_bytes(archivo.getvalue())
            with st.spinner("Generando embeddings…"):
                try:
                    rag_core.eliminar_documento(vs, f"doc_{destino.stem}")
                    n = rag_core.indexar(vs, corpus.documentos_externos(rutas=[destino]), cfg)
                except Exception as e:  # noqa: BLE001
                    st.error(rag_core.explicar_error(e, cfg))
                    st.stop()
            st.success(f"{archivo.name}: {n} fragmentos indexados.")
            st.rerun()
    with a2:
        if not resumen.empty:
            borrar = st.selectbox("Eliminar un documento", resumen.doc_id,
                                  format_func=dict(zip(resumen.doc_id, resumen.fuente)).get, key="borrar")
            if st.button("Eliminar de la base", icon=":material/delete:"):
                n = rag_core.eliminar_documento(vs, borrar)
                st.success(f"Se eliminaron {n} fragmentos.")
                st.rerun()
    with a3:
        st.markdown("Reconstruir toda la base desde el CSV, el modelo y la carpeta `docs/`.")
        if st.button("Reconstruir índice", icon=":material/sync:"):
            construir_desde_app()

# ─────────────────────────────────────────────
# 6. MONITOREO DE LA BASE DOCUMENTAL Y DEL USO
# ─────────────────────────────────────────────
with tab_mon:
    st.subheader("Estado de la base documental")
    info = rag_core.info_indice(cfg)
    frag = rag_core.listar_fragmentos(vs)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fragmentos", len(frag))
    m2.metric("Documentos", frag.doc_id.nunique() if not frag.empty else 0)
    m3.metric("Tipos de documento", frag.tipo.nunique() if not frag.empty else 0)
    m4.metric("Consultas registradas", len(rag_core.leer_consultas(cfg)))
    if info:
        st.caption(f"Última indexación: {pd.to_datetime(info['fecha']).strftime('%d/%m/%Y %H:%M')} · "
                   f"modelo de embeddings: `{info.get('embedding_model', '—')}` · "
                   f"duración: {info.get('segundos', '—')} s")
    else:
        st.caption("Sin registro de indexación: reconstruye el índice para guardarlo.")
    if info and info.get("embedding_model") != cfg.embedding_model:
        st.warning(f"El índice se creó con '{info.get('embedding_model')}' y la app usa '{cfg.embedding_model}'. "
                   "Reconstruye el índice para que las búsquedas funcionen.", icon=":material/warning:")
    if not frag.empty:
        por_tipo = (frag.assign(tipo=frag.tipo.map(lambda t: TIPOS.get(t, t)), caracteres=frag.texto.str.len())
                        .groupby("tipo", as_index=False).agg(fragmentos=("id", "count"), caracteres=("caracteres", "mean")))
        g1, g2 = st.columns(2)
        g1.altair_chart(alt.Chart(por_tipo).mark_bar(color="#0B7A70").encode(
            x=alt.X("fragmentos:Q", title="Fragmentos"),
            y=alt.Y("tipo:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            tooltip=["tipo", "fragmentos"]).properties(title="Fragmentos por tipo de documento", height=230),
            width="stretch")
        g2.altair_chart(alt.Chart(frag.assign(caracteres=frag.texto.str.len())).mark_bar(color="#0F2747").encode(
            x=alt.X("caracteres:Q", bin=alt.Bin(maxbins=20), title="Caracteres por fragmento"),
            y=alt.Y("count():Q", title="Fragmentos")).properties(title="Tamaño de los fragmentos", height=230),
            width="stretch")
        st.caption(f"Configuración de fragmentación: chunk_size={cfg.chunk_size}, chunk_overlap={cfg.chunk_overlap}. "
                   "El reglamento se divide por artículo.")

    st.subheader("Uso del chatbot")
    log = rag_core.leer_consultas(cfg)
    if log.empty:
        st.info("Todavía no hay consultas registradas. Cada pregunta que hagas en el chat aparecerá aquí.",
                icon=":material/info:")
    else:
        u1, u2, u3, u4, u5, u6 = st.columns(6)
        u1.metric("Consultas", len(log))
        u2.metric("Latencia media", f"{log.t_total_s.mean():.1f} s")
        u3.metric("Latencia p95", f"{log.t_total_s.quantile(0.95):.1f} s")
        u4.metric("Relevancia media", f"{log.relevancia_max.mean():.2f}")
        u5.metric("Con cita", f"{log.con_cita.mean():.0%}")
        u6.metric("Abstenciones", f"{log.abstencion.mean():.0%}")

        c1, c2 = st.columns(2)
        tiempos = log.reset_index().rename(columns={"index": "consulta"})
        tiempos["consulta"] += 1
        tiempos = tiempos.melt(id_vars=["consulta"], value_vars=["t_recuperacion_s", "t_generacion_s"],
                               var_name="etapa", value_name="segundos")
        tiempos["etapa"] = tiempos.etapa.map({"t_recuperacion_s": "Recuperación", "t_generacion_s": "Generación"})
        c1.altair_chart(alt.Chart(tiempos).mark_bar().encode(
            x=alt.X("consulta:O", title="Consulta"), y=alt.Y("segundos:Q", stack=True, title="Segundos"),
            color=alt.Color("etapa:N", scale=alt.Scale(domain=["Recuperación", "Generación"],
                                                       range=["#0B7A70", "#F3A712"]), title=None),
            tooltip=["consulta", "etapa", alt.Tooltip("segundos:Q", format=".2f")]
        ).properties(title="Tiempo de respuesta por consulta", height=260), width="stretch")
        citados = log.explode("documentos_citados").dropna(subset=["documentos_citados"])
        citados = citados[citados.documentos_citados != ""]
        if citados.empty:
            c2.info("Aún no hay documentos citados.")
        else:
            conteo = citados.documentos_citados.value_counts().head(10).rename_axis("documento").reset_index(name="citas")
            c2.altair_chart(alt.Chart(conteo).mark_bar(color="#0F2747").encode(
                x=alt.X("citas:Q", title="Veces citado"),
                y=alt.Y("documento:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
                tooltip=["documento", "citas"]).properties(title="Documentos más citados", height=260),
                width="stretch")

        st.markdown("**Últimas consultas**")
        ultimas = log.sort_values("fecha", ascending=False).head(20)
        st.dataframe(ultimas[["fecha", "pregunta", "t_total_s", "relevancia_max", "con_cita", "abstencion",
                              "fuentes_citadas"]].rename(columns={
                         "fecha": "Fecha", "pregunta": "Pregunta", "t_total_s": "Tiempo (s)",
                         "relevancia_max": "Relevancia", "con_cita": "Con cita", "abstencion": "Abstención",
                         "fuentes_citadas": "Fuentes citadas"}),
                     hide_index=True, width="stretch",
                     column_config={"Relevancia": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Descargar registro (CSV)", log.to_csv(index=False).encode("utf-8"),
                           "consultas_sofia.csv", "text/csv", icon=":material/download:")
        if d2.button("Borrar registro", icon=":material/delete:"):
            rag_core.borrar_consultas(cfg)
            st.rerun()
