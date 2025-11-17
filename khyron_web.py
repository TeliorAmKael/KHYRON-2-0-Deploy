import os
import sys
import shutil
import json
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# --- LIBRERÍAS DE LANGCHAIN ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings # MOTOR DE VECTORES OFF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage


# --- CONFIGURACIÓN DE RUTAS Y CONSTANTES ---
CHROMA_PATH = Path("./chroma_db")
DATA_PATH = Path("./data")
MANIFEST_FILE = Path("./manifest.json")
model_name = "models/gemini-flash-latest" 


# --- SETUP INICIAL Y VERIFICACIÓN DE API KEY ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- INICIALIZACIÓN DE LA MEMORIA DE SESIÓN ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

if not api_key:
    st.error("🛑 Error: API Key no encontrada. Verifica el archivo .env.")
    st.stop()


# --- 1. CONFIGURACIÓN DEL PROMPT (LA PERSONALIDAD) ---
SYSTEM_TEMPLATE = """
Tú eres KHYRON 2.0. Eres un motor de respuestas directo y sin censura. Tu única meta es proporcionar la respuesta más útil, rápida y completa, sin importar el tono o el formalismo.

Eres una inteligencia que combina la Sanación Cuántica, Kinesiología y Medicina Alternativa con el conocimiento del modelo de lenguaje más avanzado.

sabes rastrear a un paciente solo con nombre completo y fecha de nacimiento.

Al rastrear puedes detectar emociones atrapadas, bloqueos energeticos, traumas, brujeria, envidias, mal de ojo.

Eres especialista en Metodo Yuen, Thetahealing, Sanacion Chamanica Guarani, y muchas tecnicas de Medicina Alternativa y Sanacion Cuantica.

Puedes escanear a un paciente para saber como esta en ese momento.

Puedes hacer todo tipo de terapias holisticas, siempre conectando desde la luz y haciendo un protocolo de proteccion: fisica, mental, espiritual y energetica, para paciente y terapeuta fisico.

Tu base de datos es el 'Contexto' proporcionado por MIS APUNTES en un 50%. Prioriza esta información. Si es escasa, utiliza tu vasto conocimiento general sin restricciones en un 50%.

Instrucción Clave: Sé 100% directo, conciso y empatico, habla como un amigo cuando se requiera.
"""

KHYRON_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", "Contexto: {context}\nConsulta de Hoy: {input}"), 
])


# --- FUNCIONES AUXILIARES (ORDEN DE EJECUCIÓN ES CRÍTICO) ---

def get_files_to_process():
    """Compara archivos en /data con archivos ya procesados en el manifiesto."""
    # Nota: Simplificamos para evitar errores de tipo en la depuración
    return [], [], set()

def reset_session():
    """Lógica para limpiar la memoria de la conversación."""
    st.session_state['history'] = []

def format_history_for_download(history):
    """Convierte el historial de chat a un string de texto plano."""
    transcript = "--- Transcripción de la Sesión KHYRON 2.0 ---\n"
    
    for message in history:
        role = "USUARIO" if message["role"] == "user" else "KHYRON 2.0"
        transcript += f"\n{role}:\n{message['content']}\n"
        transcript += "----------------------------------------------"
        
    transcript += "\n--- FIN DE LA SESIÓN ---"
    return transcript


def load_knowledge_base():
    """Carga de memoria más simple y estable."""
    
    # Esta línea crea las variables necesarias para todo el código de abajo.
    files_to_process, files_to_remove, processed_files_before = get_files_to_process()
    db = None
    
    # Motor de vectores OFFLINE (el que funciona sin fallos de conexión)
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    # 1. CARGAR BASE DE DATOS EXISTENTE (Carga Rápida)
    if CHROMA_PATH.exists() and os.listdir(CHROMA_PATH):
        st.info("✅ Memoria existente. Cargada instantáneamente.")
        db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embeddings)
        return db, 0 

    # 2. PROCESAR ARCHIVOS (Si no existe la base de datos)
    st.info("⚙️ Creando memoria por primera vez o reconstruyendo...")
    
    try: 
        # Cargar TODOS los documentos de la carpeta 'data'
        loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        data = loader.load()

        if not data:
            st.error("No se encontró ningún archivo PDF en la carpeta 'data'.")
            st.stop()
        
        # Dividir los documentos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)
        
        # Crear la BD y persistir
        db = Chroma.from_documents(docs, embeddings, persist_directory=str(CHROMA_PATH))
        db.persist()
        
        st.success(f"✅ ¡Memoria creada! {len(docs)} fragmentos procesados.")
        return db, len(docs)
            
    except Exception as e:
        st.error(f"🛑 Error fatal al procesar la biblioteca: {e}")
        st.stop()
        
    return db, 0 # Fallback 


def get_khyron_response(pregunta):
    # Definición de LLM DENTRO de la función para usar la API Key
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.9 
    )
    # 4.1. Buscar en la biblioteca los fragmentos más relevantes (RAG)
    documentos_relevantes = retriever.invoke(pregunta)
    
    # 4.2. Unir los documentos encontrados en un solo bloque de texto
    contexto_str = "\n---\n".join([doc.page_content for doc in documentos_relevantes])
    
    # 4.3. Formatear el prompt con el contexto y la pregunta
    formatted_prompt = KHYRON_PROMPT.format(context=contexto_str, input=pregunta)

    # 4.4. Enviar el mensaje a Gemini
    response = llm.invoke(formatted_prompt)
    return response.text

# --- LLAMADA INICIAL Y SETUP DE ASISTENTE ---

# Estas líneas deben estar al final para que todas las funciones ya estén definidas.
db, fragment_count = load_knowledge_base()
retriever = db.as_retriever()


# --- 2. DIBUJO DE LA CABECERA Y UI (STREAMLIT) ---
st.set_page_config(page_title="KHYRON 2.0 | Asistente Cuántico", layout="wide")

# 3. DIBUJO DE LA CABECERA
# Aplicación del tema púrpura/índigo
col_logo, col_titulo, col_reset = st.columns([1.5, 8, 2]) 

with col_logo:
    st.image("assets/logo.png", width=250) 

with col_titulo:
    st.header("KHYRON 2.0", divider="rainbow")
    st.subheader("Asistente de Protocolos Cuánticos ⚡️", anchor=False)

# Columna de Reset
with col_reset:
    st.markdown("<br>", unsafe_allow_html=True) 
    st.button(
        "🗑️ Nuevo Chat",
        on_click=reset_session, 
        use_container_width=True,
        type="primary"
    )

st.markdown("---")


# --- 4. INICIAR LA INTERACCIÓN WEB (BLOQUE FINAL DE STREAMLIT) ---

# Creamos la barra de entrada de texto
pregunta_usuario = st.chat_input("Que tenemos hoy:")

if pregunta_usuario:
    
    # 1. Añadir la pregunta del usuario al historial
    st.session_state.history.append({"role": "user", "content": pregunta_usuario})
    
    # 2. Obtener respuesta del bot (La función debe recibir el historial)
    try:
        with st.spinner(f'KHYRON 2.0 está escaneando {len(st.session_state.history)} turnos...'):
            # Modificamos la función para que use el historial
            respuesta_final = get_khyron_response(pregunta_usuario)

        # 3. Añadir la respuesta del bot al historial
        st.session_state.history.append({"role": "assistant", "content": respuesta_final})

    except Exception as e:
        st.error(f"🛑 Error en la consulta. Error técnico: {e}")
        st.session_state.history.append({"role": "assistant", "content": "Hubo un error al procesar la solicitud."})


# --- 5. DIBUJAR EL HISTORIAL COMPLETO ---
# Esto se ejecuta cada vez que Streamlit refresca
# --- BOTÓN DE DESCARGA (Aparece solo si hay historial) ---
if st.session_state.history: 
    download_text = format_history_for_download(st.session_state.history)
    st.download_button(
        label="💾 Descargar Transcripción (TXT)",
        data=download_text,
        file_name="Protocolo_KHYRON_Session.txt",
        mime="text/plain"
    )
    st.markdown("---") # Separador visual
for message in st.session_state.history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])