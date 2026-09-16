# Unified Async LLM Client

Proyecto educativo en Python 3.12 con tres entregas: clientes asíncronos para OpenAI y Anthropic, extracción técnica estructurada y un sistema RAG local.

## Entrega 3: RAG local

El sistema lee documentos de `data/`, los fragmenta, los guarda en ChromaDB local y responde preguntas usando solamente los fragmentos recuperados. La respuesta final es un objeto Pydantic con texto y referencias de archivos.

Se usa `OpenAIEmbeddings` con `text-embedding-3-small` tanto al indexar como al consultar, para mantener los vectores compatibles. El LLM puede seguir siendo OpenAI o Anthropic mediante `LLM_PROVIDER`; OpenAI sigue siendo necesario para los embeddings.

## Instalación en Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración

```powershell
copy .env.example .env
```

Completa las claves necesarias y no subas `.env` a GitHub. Las variables son:

```env
OPENAI_API_KEY=tu_clave
ANTHROPIC_API_KEY=tu_clave_opcional
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=300
EMBEDDING_MODEL=text-embedding-3-small
```

La carga de `.env` está centralizada al inicio de `main.py`. Los módulos y funciones no vuelven a cargar el entorno.

## Ejecutar el RAG

```powershell
python main.py
```

En la primera ejecución se leen los archivos `.txt` y `.md` de `data/`, se crean chunks y se persisten en `vectorstore/`. Las siguientes ejecuciones detectan `vectorstore/chroma.sqlite3` y reutilizan la colección sin reindexar los documentos.

`RecursiveCharacterTextSplitter.from_tiktoken_encoder()` realiza el chunking por tokens, con chunks de 500 tokens y un overlap de 50 tokens.

La búsqueda usa similitud con `k=3`, por lo que solo se entregan tres fragmentos al prompt.

Pregunta conocida:

```text
¿Cuál es el máximo de conexiones del pool de PostgreSQL?
```

Respuesta esperada:

```json
{
  "respuesta": "El máximo es 20 conexiones.",
  "referencias": ["database_connections.txt"]
}
```

Pregunta trampa:

```text
¿Qué proveedor de pagos utiliza la API?
```

Como esa información no figura en los documentos, el prompt obliga a responder:

```json
{
  "respuesta": "No lo sé.",
  "referencias": []
}
```

La cadena LCEL aplica el prompt grounded, el LLM y `PydanticOutputParser`. Así se valida que `respuesta` no esté vacía y que las referencias sean nombres de archivos `.txt` o `.md`.

## Tests

```powershell
python -m pytest
```

Los tests RAG usan embeddings, vector store y LLM falsos; no realizan llamadas a OpenAI ni Anthropic.

## Estructura principal

```text
unified-async-llm-client/
├── data/                  # Dataset técnico de ejemplo
├── vectorstore/           # Se crea localmente y está ignorado por Git
├── src/
│   ├── pipeline/          # Entrega 2
│   └── rag/
│       ├── ingestion.py   # Lectura, chunks y Chroma persistente
│       ├── retriever.py   # Similarity search con k=3
│       ├── prompt.py      # Prompt grounded y PydanticOutputParser
│       └── chain.py       # get_rag_response() asíncrona
├── tests/
│   └── test_rag.py
├── main.py
└── requirements.txt
```

Los clientes y tests de las entregas 1 y 2 se conservan en `src/` y `tests/`.
