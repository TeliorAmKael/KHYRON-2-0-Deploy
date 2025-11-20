import os
import sys
import shutil
import zipfile
import gdown
import gc
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# --- LIBRERÍAS LIGERAS (Solo Google) ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- CONFIGURACIÓN ---
CHROMA_PATH = Path("chroma_db")

# ⚠️⚠️ ATENCIÓN AQUÍ: Pega tu ID dentro de las comillas rojas ⚠️⚠️
# Debe quedar así: ID_DRIVE_ZIP = "1AbCdEfG..."
ID_DRIVE_ZIP = "PEGA_AQUI_TU_NUEVO_ID_DEL_DRIVE"

# Modelos (Deben coincidir con lo que generaste)
EMBEDDING_MODEL = "models/text-embedding-004" 
CHAT_MODEL = "gemini-1.5-flash"

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="KHYRON 2.0", layout="wide")

# --- GESTIÓN DE MEMORIA (CACHEADA Y LIGERA) ---
@st.cache_resource(show_spinner=False)
def setup_knowledge_base():
    """Descarga y conecta la memoria. Solo se ejecuta una vez."""
    
    # 1. Configurar Embeddings (El mismo modelo que usaste en local)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, 
        google_api_key=api_key
    )
    
    # 2. Si no existe la carpeta, descargar
    if not (CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir())):
        print("⬇️ Descargando memoria ligera...")
        try:
            output_zip = "chroma_db.zip"
            url = f'https://drive.google.com/uc?id={ID_DRIVE_ZIP}'
            gdown.download(url, output_zip, quiet=False)
            
            print("📦 Descomprimiendo...")
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(".") 
            
            # Limpieza agresiva de RAM
            if os.path.exists(output_zip):
                os.remove(output_zip)
            gc.collect()
            
        except Exception as e:
            st.error(f"Error descarga: {e}")
            return None

    print("✅ Memoria lista.")
    return Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embeddings)

# --- CHAT ---
def get_response(query, db):
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.7)
    
    # Recuperar solo 3 fragmentos para ser rápido y ahorrar RAM
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    
    # Cuidado con las comillas aquí (ya corregido)
    prompt = f"""Eres KHYRON 2.0, experto en Sanación Cuántica.
    Responde directo usando este contexto:
    {context}
    
    Pregunta: {query}"""
    
    return llm.invoke(prompt).content

# --- INTERFAZ ---
if not api_key:
    st.error("Falta API Key. Configura GOOGLE_API_KEY en Render.")
    st.stop()

with st.spinner('Conectando con la nube cuántica...'):
    db = setup_knowledge_base()

st.title("KHYRON 2.0 ⚡️")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("¿En qué te ayudo?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    if db:
        try:
            response = get_response(prompt, db)
            st.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Error: La memoria no está conectada.")