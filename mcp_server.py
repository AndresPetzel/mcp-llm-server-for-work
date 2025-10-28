# --- Importaciones Principales ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
import httpx
from typing import Any, Dict
import os

# --- Importaciones de LLMs y APIs ---
import google.generativeai as genai
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# --- Importaciones para Google Secret Manager ---
from google.cloud import secretmanager
import google.auth

# --------------------------------------------------------------------------
# 0. FUNCIÓN PARA OBTENER SECRETOS DE GOOGLE SECRET MANAGER
# --------------------------------------------------------------------------
def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str | None:
    """Obtiene el valor de un secreto desde Google Secret Manager."""
    # --- ¡CORRECCIÓN AQUÍ! ---
    # Comprueba si el project_id está vacío o si *todavía* es el placeholder
    if not project_id or project_id == "PON_AQUI_TU_GCP_PROJECT_ID":
    # --- FIN DE LA CORRECCIÓN ---
         print(f"ERROR CRÍTICO: GCP_PROJECT_ID ('{project_id}') no parece estar configurado correctamente en el código.")
         return None
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except google.auth.exceptions.DefaultCredentialsError as e:
        print(f"ERROR: Credenciales de GCP no encontradas o inválidas: {e}. "
              "Verifica rol 'Secret Manager Secret Accessor'.")
        return None
    except Exception as e:
        print(f"ERROR: No se pudo obtener el secreto '{secret_id}' del proyecto '{project_id}': {e}")
        return None

# --- TU PROJECT ID (¡YA LO PUSISTE CORRECTAMENTE!) ---
GCP_PROJECT_ID = "ensayo-de-automatizacion-ap"


# --------------------------------------------------------------------------
# 1. CREAR LAS APLICACIONES
# --------------------------------------------------------------------------
app = FastAPI(
    title="Mi Servidor MCP Unificado para LLMs",
    description="Un servidor HTTP que expone una herramienta genérica para llamar a varios LLMs."
)
mcp = FastMCP("llm-unified-tools")

# --------------------------------------------------------------------------
# 2. CONFIGURACIÓN DE PROVEEDORES Y API KEYS (Usando Secret Manager)
# --------------------------------------------------------------------------
providers_config = {
    "groq": {
        "api_key": get_secret(GCP_PROJECT_ID, "GROQ_API_KEY"), # Asegúrate que este sea tu ID de secreto real
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openrouter": {
        "api_key": get_secret(GCP_PROJECT_ID, "OPENROUTER_API_KEY"), # Asegúrate que este sea tu ID de secreto real
        "base_url": "https://openrouter.ai/api/v1",
    },
    "gemini": {
        "api_key": get_secret(GCP_PROJECT_ID, "GEMINI_API_KEY"), # Asegúrate que este sea tu ID de secreto real
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", # OJO: URL por verificar
    },
    "openai": {
        "api_key": get_secret(GCP_PROJECT_ID, "OPENAI_API_KEY"), # Asegúrate que este sea tu ID de secreto real
        "base_url": None,
    },
    "perplexity": {
        "api_key": get_secret(GCP_PROJECT_ID, "PERPLEXITY_API_KEY"), # Asegúrate que este sea tu ID de secreto real
        "base_url": "https://api.perplexity.ai",
    }
    # Añade aquí otros proveedores
}

# --- Configuración Específica (Opcional) ---
gemini_api_key_direct = get_secret(GCP_PROJECT_ID, "GEMINI_API_KEY") # Usa el mismo ID
if gemini_api_key_direct:
    try: genai.configure(api_key=gemini_api_key_direct)
    except Exception as e: print(f"Error al configurar genai: {e}")
else: pass

# Necesitarás crear un secreto llamado 'anthropic-api-key-secret-id' o cambiar el nombre aquí
anthropic_api_key_direct = get_secret(GCP_PROJECT_ID, "anthropic-api-key-secret-id")
if anthropic_api_key_direct:
    try: anthropic_client_direct = AsyncAnthropic(api_key=anthropic_api_key_direct)
    except Exception as e: print(f"Error al configurar Anthropic: {e}"); anthropic_client_direct = None
else: anthropic_client_direct = None


# --------------------------------------------------------------------------
# 3. DEFINICIÓN DE LA HERRAMIENTA MCP Y ENDPOINT HTTP
# --------------------------------------------------------------------------

@mcp.tool()
async def call_llm(
    provider_name: str,
    prompt: str,
    model_name: str = "default",
    system_message: str = "Eres un asistente útil.",
    temperature: float = 0.7
) -> str:
    """Llama a un LLM compatible con API OpenAI."""
    provider_name = provider_name.lower() # Asegura minúsculas
    if provider_name not in providers_config or not providers_config[provider_name].get("api_key"):
        api_key_value = providers_config.get(provider_name, {}).get("api_key")
        if not api_key_value:
             # Este print es útil para depurar si falla get_secret()
             print(f"ADVERTENCIA INTERNA: API Key para '{provider_name}' no fue obtenida de Secret Manager o no está configurada en providers_config.")
             raise ValueError(f"Proveedor '{provider_name}' no válido o API key no configurada/accesible.")

    selected_provider = providers_config[provider_name]
    api_key = selected_provider["api_key"]
    base_url = selected_provider["base_url"]

    # Determinar modelo
    if model_name == "default":
        if provider_name == "groq": model_to_use = "llama-3.1-8b-instant"
        elif provider_name == "openrouter": model_to_use = "mistralai/mistral-7b-instruct:free"
        elif provider_name == "gemini": model_to_use = "gemini-pro"
        elif provider_name == "openai": model_to_use = "gpt-4o-mini"
        elif provider_name == "perplexity": model_to_use = "llama-3.1-sonar-small-128k-online"
        else: raise ValueError(f"Modelo por defecto no definido para '{provider_name}'.")
    else:
        model_to_use = model_name

    # Configurar cliente OpenAI
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
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content or f"Respuesta vacía de {provider_name}."
        else:
            return f"Error: No se recibió respuesta válida de {provider_name}."
    except Exception as e:
        print(f"Error detallado llamando a {provider_name} ({model_to_use}): {e}")
        error_detail = str(e)
        if "authentication" in error_detail.lower():
             raise HTTPException(status_code=401, detail=f"Error de autenticación con {provider_name}. Verifica la API Key.")
        elif "rate limit" in error_detail.lower():
             raise HTTPException(status_code=429, detail=f"Límite de peticiones excedido con {provider_name}.")
        else:
             raise HTTPException(status_code=503, detail=f"Error al contactar al proveedor {provider_name}: {error_detail}")
    finally:
        await client.close()

# --- Modelo Pydantic (sin cambios) ---
class LLMRequestPayload(BaseModel):
    provider_name: str
    prompt: str
    model_name: str = "default"
    system_message: str = "Eres un asistente útil."
    temperature: float = 0.7

# --- Endpoint HTTP (sin cambios) ---
@app.post("/tools/call_llm", summary="Llamar a un LLM genérico")
async def http_call_llm(payload: LLMRequestPayload):
    """Endpoint HTTP genérico para llamar a LLMs (API compatible OpenAI)."""
    try:
        result = await call_llm(
            provider_name=payload.provider_name,
            prompt=payload.prompt,
            model_name=payload.model_name,
            system_message=payload.system_message,
            temperature=payload.temperature
        )
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error inesperado en http_call_llm: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")


# --- Ruta Raíz (sin cambios) ---
@app.get("/")
def read_root():
    return {"message": "Servidor MCP Unificado para LLMs está funcionando (con Secret Manager)."}

# --- Opcional: Montar el router MCP ---
# app.include_router(mcp.router, prefix="/mcp")