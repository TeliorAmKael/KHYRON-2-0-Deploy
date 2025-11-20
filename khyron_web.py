import os
import sys
import shutil
import zipfile
import gdown
import gc
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# --- LIBRERÍAS (FAISS + GOOGLE) ---
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURACIÓN ---
FAISS_PATH = "faiss_index"
# Tu ID del zip de Drive
ID_DRIVE_ZIP = "1n9IhMfnqHzHogAi0FdOpxAnxMoqB7lux"

# Modelos (Deben coincidir con los que usaste para generar)
EMBEDDING_MODEL = "models/text-embedding-004" 
CHAT_MODEL = "gemini-1.5-flash"

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="KHYRON 2.0", layout="wide")

# --- GESTIÓN DE MEMORIA (FAISS - SIN SQLITE) ---
@st.cache_resource(show_spinner=False)
def setup_knowledge_base():
    # 1. Configurar Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, 
        google_api_key=api_key
    )
    
    # 2. Descargar si no existe la carpeta
    if not os.path.exists(FAISS_PATH):
        print("⬇️ Descargando cerebro FAISS...")
        try:
            output_zip = "faiss_index.zip"
            url = f'https://drive.google.com/uc?id={ID_DRIVE_ZIP}'
            gdown.download(url, output_zip, quiet=False)
            
            print("📦 Descomprimiendo...")
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(".") 
            
            # Limpieza
            if os.path.exists(output_zip):
                os.remove(output_zip)
            gc.collect()
            
        except Exception as e:
            st.error(f"Error descarga: {e}")
            return None
            
    # 3. Cargar índice FAISS
    # (allow_dangerous=True es seguro aquí porque nosotros creamos el archivo)
    try:
        vector_store = FAISS.load_local(
            FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ Cerebro FAISS cargado.")
        return vector_store
    except Exception as e:
        st.error(f"Error cargando FAISS: {e}")
        return None

# --- CHAT ---
def get_response(query, db):
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.7)
    
    # Buscamos los 3 fragmentos más parecidos
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    
    prompt_text = f"""Eres KHYRON 2.0, experto en Sanación Cuántica.
    Responde directo usando este contexto:
    {context}
    
    Pregunta: {query}"""
    
    return llm.invoke(prompt_text).content

# --- INTERFAZ ---
if not api_key:
    st.error("Falta API Key. Configura GOOGLE_API_KEY en Render.")
    st.stop()

with st.spinner('Iniciando sistemas FAISS...'):
    db = setup_knowledge_base()

st.title("KHYRON 2.0 ⚡️ (FAISS Core)")

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
        st.error("Error: El cerebro no está conectado.")