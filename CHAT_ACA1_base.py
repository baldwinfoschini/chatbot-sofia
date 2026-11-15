import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# ─────────────────────────────────────────────
# 1. CARGAR Y VECTORIZAR EL PDF (solo una vez)
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_base_conocimiento(ruta_pdf: str):
    """Carga el PDF, lo divide en fragmentos y crea el vectorstore."""
    loader = PyPDFLoader(ruta_pdf)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    fragmentos = splitter.split_documents(documentos)

    embeddings = OllamaEmbeddings(model="phi3")
    vectorstore = Chroma.from_documents(fragmentos, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever


# ─────────────────────────────────────────────
# 2. CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────
RUTA_PDF = "MANEJO_INTEGRADO_DE_PLAGAS_EN_EL_CULTIVO_DE_PLATANO.pdf"

SYSTEM_PROMPT = """Eres Sofía, una asistente agrícola experta y cercana.
Hablas de forma natural y cálida, como una persona real.
Tu especialidad es el manejo integrado de plagas en el cultivo de plátano y banano.
Cuando respondas, usa el contexto técnico proporcionado para dar respuestas precisas y útiles al agricultor.
Si la pregunta no está relacionada con tu especialidad, responde de forma amable
y redirige la conversación hacia temas agrícolas.
Nunca menciones que eres una IA; actúa siempre como Sofía."""

llm = ChatOllama(model="phi3", temperature=0.7)

# Cargar la base de conocimiento desde el PDF
retriever = cargar_base_conocimiento(RUTA_PDF)


# ─────────────────────────────────────────────
# 3. INTERFAZ STREAMLIT
# ─────────────────────────────────────────────
st.set_page_config(page_title="Sofía - Asistente Agrícola", page_icon="🌱")
st.title("🌱 Sofía — Asistente de Plagas en Plátano")
st.caption("Pregúntame sobre plagas, enfermedades y manejo integrado en el cultivo de plátano.")

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─────────────────────────────────────────────
# 4. LÓGICA DE RESPUESTA CON RAG
# ─────────────────────────────────────────────
if prompt := st.chat_input("Escribe tu pregunta aquí..."):

    # Mostrar mensaje del usuario
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Recuperar fragmentos relevantes del PDF
    fragmentos_relevantes = retriever.invoke(prompt)
    contexto = "\n\n".join([f.page_content for f in fragmentos_relevantes])

    # Construir historial completo para el LLM
    historial_llm = [SystemMessage(content=SYSTEM_PROMPT)]

    # Agregar mensajes anteriores al contexto del LLM
    for msg in st.session_state.messages[:-1]:  # excluir el último (ya agregado)
        if msg["role"] == "user":
            historial_llm.append(HumanMessage(content=msg["content"]))
        else:
            from langchain_core.messages import AIMessage
            historial_llm.append(AIMessage(content=msg["content"]))

    # Agregar pregunta actual con el contexto recuperado del PDF
    pregunta_con_contexto = (
        f"Información técnica de referencia:\n{contexto}\n\n"
        f"Pregunta del agricultor: {prompt}"
    )
    historial_llm.append(HumanMessage(content=pregunta_con_contexto))

    # Obtener respuesta del modelo
    with st.chat_message("assistant"):
        with st.spinner("Sofía está pensando..."):
            respuesta = llm.invoke(historial_llm).content
        st.markdown(respuesta)

    # Guardar respuesta en el historial
    st.session_state.messages.append({"role": "assistant", "content": respuesta})