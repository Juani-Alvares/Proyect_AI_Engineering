# Unified Async LLM Client

Cliente educativo en Python 3.12 para trabajar con **OpenAI y Anthropic mediante una interfaz común y asíncrona**.

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

`.env` está incluido en `.gitignore`. **Nunca subas las claves a GitHub.**

## Ejecutar

```powershell
python main.py
```

El programa realiza dos pruebas con el mismo mensaje:

1. `generate()` devuelve la respuesta completa.
2. `stream()` imprime los fragmentos a medida que llegan.

Si no hay saldo/cuota o la API devuelve un error, se muestra un mensaje controlado. Los tests no hacen llamadas reales.

## Tests sin gastar dinero

```powershell
python -m pytest
```

Los tests comprueban validaciones, selección de proveedor y comportamiento ante claves ausentes. No necesitan API keys ni saldo.

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
│   └── manager.py
└── tests/
    ├── __init__.py
    ├── test_schema.py
    ├── test_clients.py
    └── test_manager.py
```
