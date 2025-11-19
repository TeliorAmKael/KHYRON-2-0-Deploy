import os
import sys
import shutil
import json
import zipfile
import gdown
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# --- LIBRERÍAS ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

# --- CONFIGURACIÓN ---
CHROMA_PATH = Path("chroma_db")
# ⚠️ ¡PEGA AQUÍ TU NUEVO ID DEL ZIP! ⚠️
ID_DRIVE_ZIP = "https://drive.google.com/file/d/1Z6qm8SOqXkajMkKfZ4YvuIPOFxTT9YKR/view?usp=drive_link" 

# CORRECCIÓN CRÍTICA: El nombre sin "models/"
model_name = "gemini-1.5-flash" 

# --- API KEY ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- MEMORIA DE SESIÓN ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- PROMPT ---
SYSTEM_TEMPLATE = """
Tú eres KHYRON 2.0. Eres un motor de respuestas directo y sin censura. 
Eres una inteligencia que combina la Sanación Cuántica, Kinesiología y Medicina Alternativa.
Instrucción Clave: Sé 100% directo, conciso y empatico.
"""

KHYRON_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", "Contexto: {context}\nConsulta de Hoy: {input}"), 
])

# --- FUNCIONES ---

def reset_session():
    st.session_state['history'] = []

def format_history_for_download(history):
    transcript = "--- KHYRON 2.0 ---\n"
    for message in history:
        role = "User" if message["role"] == "user" else "KHYRON"
        transcript += f"\n{role}: {message['content']}\n"
    return transcript

def load_knowledge_base():
    """Carga HÍBRIDA: Descarga de Drive si no existe localmente."""
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = None

    # 1. ¿Existe localmente?
    if CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir()):
        print("✅ Memoria detectada en disco.")
        db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embeddings)
        return db

    # 2. Si no, descargar de Nube
    st.info("⬇️ Descargando memoria maestra desde la nube...")
    try:
        output_zip = "chroma_db.zip"
        url = f'https://drive.google.com/uc?id={ID_DRIVE_ZIP}'
        gdown.download(url, output_zip, quiet=False)
        
        st.info("📦 Descomprimiendo...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(".") 
            
        if os.path.exists(output_zip):
            os.remove(output_zip)
            
        st.success("✅ Memoria instalada.")
        db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embeddings)
        return db
            
    except Exception as e:
        st.error(f"🛑 Error descargando memoria: {e}")
        st.stop()
        return None

def get_khyron_response(pregunta):
    # LLM Configurado correctamente
    llm = ChatGoogleGenerativeAI(
        model=model_name, # gemini-1.5-flash
        google_api_key=api_key,
        temperature=0.9 
    )
    
    documentos_relevantes = retriever.invoke(pregunta)
    contexto_str = "\n---\n".join([doc.page_content for doc in documentos_relevantes])
    formatted_prompt = KHYRON_PROMPT.format(context=contexto_str, input=pregunta)
    
    response = llm.invoke(formatted_prompt)
    return response.content

# --- INICIO ---
db = load_knowledge_base()
retriever = db.as_retriever()

# --- INTERFAZ ---
st.set_page_config(page_title="KHYRON 2.0", layout="wide")

st.header("KHYRON 2.0 ⚡️")
st.button("🗑️ Nuevo Chat", on_click=reset_session, type="primary")
st.markdown("---")

pregunta_usuario = st.chat_input("Que tenemos hoy:")

if pregunta_usuario:
    st.session_state.history.append({"role": "user", "content": pregunta_usuario})
    try:
        with st.spinner('Analizando...'):
            respuesta = get_khyron_response(pregunta_usuario)
        st.session_state.history.append({"role": "assistant", "content": respuesta})
    except Exception as e:
        st.error(f"Error: {e}")

if st.session_state.history: 
    st.download_button("💾 Guardar Chat", format_history_for_download(st.session_state.history), "chat.txt")

for message in st.session_state.history:
    st.chat_message(message["role"]).write(message["content"])