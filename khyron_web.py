# --- PARCHE DE SQLITE PARA RENDER (OBLIGATORIO AL INICIO) ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -----------------------------------------------------------

import os
import shutil
import zipfile
import gdown
import gc
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# --- LIBRERÍAS ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURACIÓN ---
CHROMA_PATH = Path("chroma_db")
# ⚠️ ASEGÚRATE DE QUE TU ID ESTÉ AQUÍ ⚠️
ID_DRIVE_ZIP = "https://drive.google.com/file/d/199gAEBOMibvOzzAI1Yqt5fIuw9J-34z1/view?usp=drive_link"

EMBEDDING_MODEL = "models/text-embedding-004" 
CHAT_MODEL = "gemini-1.5-flash"

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="KHYRON 2.0", layout="wide")

# --- GESTIÓN DE MEMORIA ---
@st.cache_resource(show_spinner=False)
def setup_knowledge_base():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    
    if not (CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir())):
        print("⬇️ Descargando memoria...")
        try:
            output_zip = "chroma_db.zip"
            url = f'https://drive.google.com/uc?id={ID_DRIVE_ZIP}'
            gdown.download(url, output_zip, quiet=False)
            
            print("📦 Descomprimiendo...")
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(".") 
            
            if os.path.exists(output_zip):
                os.remove(output_zip)
            gc.collect()
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    return Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embeddings)

# --- CHAT ---
def get_response(query, db):
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.7)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Eres KHYRON 2.0. Responde con esto:\n{context}\n\nPregunta: {query}"
    return llm.invoke(prompt).content

# --- INTERFAZ ---
if not api_key:
    st.error("Falta API Key")
    st.stop()

# Carga silenciosa
try:
    db = setup_knowledge_base()
except Exception as e:
    st.error(f"Error crítico DB: {e}")
    db = None

st.title("KHYRON 2.0 ⚡️")

if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if db:
        try:
            response = get_response(prompt, db)
            st.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
             st.error("Error de comunicación.")