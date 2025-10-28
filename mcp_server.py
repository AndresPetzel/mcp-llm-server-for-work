# --- Importaciones Principales ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # Importamos BaseModel
from mcp.server.fastmcp import FastMCP
import httpx # Aunque no lo usemos directamente ahora, es bueno tenerlo
from typing import Any, Dict
import os # Para leer variables de entorno (como API keys)

# --- Importaciones de LLMs y APIs ---

# Google Gemini (Importado por precaución)
import google.generativeai as genai

# OpenAI (ChatGPT - ¡Esta es la que usaremos principalmente!)
from openai import AsyncOpenAI

# Anthropic (Claude - Importado por precaución)
from anthropic import AsyncAnthropic

# --------------------------------------------------------------------------
# 1. CREAR LAS APLICACIONES
# --------------------------------------------------------------------------
app = FastAPI(
    title="Mi Servidor MCP Unificado para LLMs",
    description="Un servidor HTTP que expone una herramienta genérica para llamar a varios LLMs compatibles con API OpenAI."
)
mcp = FastMCP("llm-unified-tools")

# --------------------------------------------------------------------------
# 2. CONFIGURACIÓN DE PROVEEDORES Y API KEYS
# --------------------------------------------------------------------------
# Define la estructura de proveedores (puedes añadir más)
providers_config = {
    "groq": {
        "api_key": os.environ.get("GROQ_API_KEY"),
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openrouter": {
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
    },
    "gemini": {
        "api_key": os.environ.get("GEMINI_API_KEY"),
        # Verifica la URL compatible con OpenAI correcta en la documentación de Google
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", # OJO: Esta URL puede requerir ajustes
    },
    "openai": {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "base_url": None, # Usa la URL por defecto de OpenAI
    }
    # Añade aquí otros proveedores compatibles con API OpenAI
}

# Validar que al menos las claves estén presentes
for name, config in providers_config.items():
    if not config["api_key"]:
        print(f"ADVERTENCIA: API Key para '{name}' no encontrada en las variables de entorno.")

# Configuración específica de Gemini (si la usaras directamente - OPCIONAL)
# gemini_api_key_direct = os.environ.get("GEMINI_API_KEY")
# if gemini_api_key_direct:
#     genai.configure(api_key=gemini_api_key_direct)

# Configuración específica de Anthropic (si la usaras directamente - OPCIONAL)
# anthropic_api_key_direct = os.environ.get("ANTHROPIC_API_KEY")
# if anthropic_api_key_direct:
#     anthropic_client_direct = AsyncAnthropic(api_key=anthropic_api_key_direct)
# else:
#     anthropic_client_direct = None


# --------------------------------------------------------------------------
# 3. DEFINICIÓN DE LA HERRAMIENTA MCP Y ENDPOINT HTTP
# --------------------------------------------------------------------------

# --- Herramienta Genérica para LLMs (usando API compatible OpenAI) ---
@mcp.tool()
async def call_llm(
    provider_name: str,
    prompt: str,
    model_name: str = "default",
    system_message: str = "Eres un asistente útil.",
    temperature: float = 0.7
) -> str:
    """
    Llama a un LLM a través de un proveedor específico compatible con la API de OpenAI.
    """
    if provider_name not in providers_config or not providers_config[provider_name].get("api_key"):
        # Usamos ValueError aquí porque es un error de lógica interna, no de petición HTTP
        raise ValueError(f"Proveedor '{provider_name}' no válido o API key no configurada.")

    selected_provider = providers_config[provider_name]
    api_key = selected_provider["api_key"]
    base_url = selected_provider["base_url"]

    # Determinar modelo
    if model_name == "default":
        # Establecer modelos por defecto (ejemplos)
        if provider_name == "groq": model_to_use = "llama3-8b-8192"
        elif provider_name == "openrouter": model_to_use = "mistralai/mistral-7b-instruct:free"
        elif provider_name == "gemini": model_to_use = "gemini-pro" # Verificar compatibilidad
        elif provider_name == "openai": model_to_use = "gpt-4o-mini"
        else: raise ValueError(f"Modelo por defecto no definido para '{provider_name}'.")
    else:
        model_to_use = model_name

    # Configurar cliente OpenAI para el proveedor
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        response = await client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        # Extraer respuesta
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content or f"Respuesta vacía de {provider_name}."
        else:
            return f"Error: No se recibió respuesta válida de {provider_name}."
    # Capturamos Exception genérica aquí, pero levantamos HTTPException para que FastAPI lo maneje
    except Exception as e:
        print(f"Error detallado llamando a {provider_name} ({model_to_use}): {e}") # Log para nosotros
        # Devolvemos un error HTTP claro al cliente (n8n)
        raise HTTPException(status_code=503, detail=f"Error al contactar al proveedor {provider_name}: {str(e)}")
    finally:
        # Aseguramos cerrar el cliente asíncrono
        await client.close()

# --- Modelo Pydantic para la entrada ---
class LLMRequestPayload(BaseModel):
    provider_name: str
    prompt: str
    model_name: str = "default" # Valor por defecto
    system_message: str = "Eres un asistente útil." # Valor por defecto
    temperature: float = 0.7 # Valor por defecto

# --- Endpoint HTTP para n8n (¡VERSIÓN CORREGIDA!) ---
@app.post("/tools/call_llm", summary="Llamar a un LLM genérico")
async def http_call_llm(payload: LLMRequestPayload): # Usa el modelo Pydantic
    """
    Endpoint HTTP genérico para llamar a LLMs (API compatible OpenAI).
    Espera JSON que coincida con LLMRequestPayload.
    Devuelve: {"result": "Texto generado"} o {"detail": "Mensaje de error"}
    """
    # El bloque try/except ahora está aquí, envolviendo la llamada a call_llm
    try:
        # Llama a la función principal con los datos validados por Pydantic
        result = await call_llm(
            provider_name=payload.provider_name,
            prompt=payload.prompt,
            model_name=payload.model_name,
            system_message=payload.system_message,
            temperature=payload.temperature
        )
        return {"result": result}
    except ValueError as e: # Captura errores de validación de call_llm (ej. proveedor inválido)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e: # Re-lanza errores HTTP que ocurrieron dentro de call_llm
        raise e
    except Exception as e: # Captura cualquier otro error inesperado
        print(f"Error inesperado en http_call_llm: {e}") # Log del error en el servidor
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")


# --- Ruta Raíz ---
@app.get("/")
def read_root():
    return {"message": "Servidor MCP Unificado para LLMs está funcionando."}

# --- Opcional: Si necesitaras usar las bibliotecas específicas directamente ---
# Podrías añadir otras funciones @mcp.tool() y @app.post() aquí
# que usen 'genai' o 'anthropic_client_direct' si la API compatible
# no fuera suficiente para alguna tarea muy específica.

# --- Opcional: Montar el router MCP si quieres exponer las definiciones ---
# Si un cliente MCP necesita ver la *definición* de las herramientas
# (no solo llamarlas por HTTP), podrías necesitar esto. Pruébalo sin esto primero.
# app.include_router(mcp.router, prefix="/mcp")