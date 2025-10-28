# 1. Usar una imagen base ligera de Python (3.11 es compatible con las librerías)
FROM python:3.11-slim

# 2. Instalar 'uv' (el gestor de paquetes moderno para Python)
# Usamos pip aquí solo para instalar uv mismo.
RUN pip install uv

# 3. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /usr/src/app

# 4. Instalar las dependencias de Python usando uv
#    Incluimos:
#    - mcp[cli]: El SDK base de MCP.
#    - httpx: Para hacer llamadas HTTP (aunque las librerías de LLM lo usan internamente).
#    - uvicorn: El servidor ASGI que ejecutará FastAPI.
#    - fastapi: El framework web para crear los endpoints HTTP.
#    - openai: La librería para interactuar con APIs compatibles con OpenAI (incluyendo Groq, OpenRouter, etc.).
#    - google-generativeai: La librería específica para Google Gemini.
#    - anthropic: La librería específica para Anthropic Claude.
#    (Añade aquí otras librerías si las necesitas, ej. para Manus.im o Google Cloud)
RUN uv pip install --system "mcp[cli]" httpx uvicorn fastapi openai google-generativeai anthropic packaging pydantic google-cloud-secret-manager google-auth

# 5. Copiar tu script de python (mcp_server.py) y cualquier otro archivo necesario
#    desde tu carpeta local al directorio de trabajo dentro del contenedor.
COPY . .

# 6. Exponer el puerto 8080 (el puerto en el que uvicorn escuchará dentro del contenedor)
EXPOSE 8080

# 7. El comando para arrancar el servidor web cuando el contenedor se inicie.
#    Le dice a uvicorn que ejecute el objeto 'app' (la instancia de FastAPI)
#    que se encuentra dentro del archivo 'mcp_server.py'.
#    --host 0.0.0.0 : Hace que el servidor sea accesible desde fuera del contenedor (importante para Docker).
#    --port 8080 : El puerto en el que escuchará.
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8080"]