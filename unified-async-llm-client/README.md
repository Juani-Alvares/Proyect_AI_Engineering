# Unified Async LLM Client

Cliente educativo en Python 3.12 para trabajar con **OpenAI y Anthropic mediante una interfaz común y asíncrona**.

## Entrega 2: Pipeline de extracción técnica

La segunda entrega agrega un pipeline LCEL que recibe un párrafo técnico y devuelve un objeto Pydantic con tecnologías, nivel de criticidad (`baja`, `media` o `alta`) y un resumen técnico. La cadena usa `ChatPromptTemplate | model.with_structured_output(TechnicalExtraction)` y aplica hasta dos intentos automáticos mediante `.with_retry()` ante errores de JSON, parseo o validación estructurada.

La salida de `with_structured_output()` es directamente el objeto Pydantic; por eso el pipeline valida campos faltantes o inválidos, pero no consulta `finish_reason` de forma directa. Ese metadato depende de la respuesta cruda que exponga cada proveedor y LangChain lo abstrae en este flujo simple.

## Qué demuestra

- Pydantic para validar mensajes y configuración.
- `async`/`await` para llamadas no bloqueantes.
- Streaming con generadores asíncronos (`async for` + `yield`).
- Una interfaz abstracta común para distintos proveedores.
- `AsyncLLMManager` para seleccionar OpenAI o Anthropic.
- Variables de entorno para no guardar claves en el código.
- Manejo de errores de autenticación, cuota/límite y conexión.
- Tests locales que no consumen API ni requieren saldo.

## Instalación en Windows

Desde la carpeta del proyecto:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración

```powershell
copy .env.example .env
```

Edita `.env` con **solo la clave del proveedor que vayas a usar**.

OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=tu_clave
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=300
```

Anthropic:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=tu_clave
LLM_MODEL=claude-sonnet-5
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=300
```

`.env` está incluido en `.gitignore`. 

## Ejecutar el ejemplo de Entrega 2

```powershell
python main.py
```

El programa procesa este texto:

```text
La API está desarrollada con FastAPI, utiliza Redis como caché y PostgreSQL como base de datos. Se detectaron problemas de conexiones concurrentes y aumento de latencia.
```

Salida esperada (el contenido exacto depende del modelo):

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Se detectaron problemas de concurrencia y latencia."
}
```

El proveedor se elige mediante `LLM_PROVIDER` en `.env`. Se conservan los clientes asíncronos y el streaming de la Entrega 1. Si falta una clave o la API falla, `main.py` muestra un error controlado.

## Tests sin gastar dinero

```powershell
python -m pytest
```

Los tests comprueban validaciones, selección de proveedor, comportamiento ante claves ausentes, el pipeline asíncrono con un mock y un reintento LCEL. No necesitan API keys ni saldo.

## Estructura

```text
unified-async-llm-client/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── schemas.py
│   ├── base_client.py
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── manager.py
│   └── pipeline/
│       ├── prompt.py      # ChatPromptTemplate modular
│       └── chain.py       # Cadena LCEL y process_text()
└── tests/
    ├── __init__.py
    ├── test_schema.py
    ├── test_clients.py
    ├── test_manager.py
    └── test_pipeline.py
```
