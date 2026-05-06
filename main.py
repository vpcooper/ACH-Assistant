import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# 1. Cargar configuración
load_dotenv()
api_key = os.getenv("API_KEY_GEMINI")

if not api_key:
    raise ValueError("¡Falta la API_KEY_GEMINI en el archivo .env!")

# 2. Configurar el "Cerebro" (LLM) y los "Vectores" (Embeddings)
# Usamos Flash por ser rápido y gratuito
llm = Gemini(model="models/gemini-3-flash-preview", api_key=api_key)
embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=api_key)

# Definir la configuración global
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024  # Tamaño de los trozos de texto

# 3. Gestión de la Memoria (Persistencia)
PERSIST_DIR = "./storage"
DATA_DIR = "./data"

def inicializar_sistema():
    if not os.path.exists(PERSIST_DIR):
        print("— Creando nuevo índice (esto puede tardar según el tamaño de los PDF)...")
        # Lee todos los archivos en /data/manuales y /data/tickets
        documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(documents)
        # Guarda el índice para la próxima vez
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("— Cargando índice existente desde la carpeta /storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

# 4. Flujo principal
if __name__ == "__main__":
    # Inicializar
    index = inicializar_sistema()
    query_engine = index.as_query_engine(similarity_top_k=5) # Busca los 5 mejores fragmentos

    print("\n--- ASISTENTE ACH LISTO ---")
    print("Escribe 'salir' para terminar.\n")

    while True:
        pregunta = input("Tu consulta: ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            break
        
        # Consultar a la IA
        print("\nAnalizando manuales y tickets...")
        respuesta = query_engine.query(pregunta)
        
        print(f"\nRespuesta:\n{respuesta}\n")
        print("-" * 30)