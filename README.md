# Unified Async LLM Client / RAG

## Qué hace el proyecto

Proyecto educativo en Python 3.12 dividido en cuatro entregas. La Entrega 1 implementa clientes asíncronos para OpenAI y Anthropic con streaming. La Entrega 2 agrega extracción de entidades técnicas con LangChain y Pydantic. La Entrega 3 implementa un sistema RAG local sobre documentos técnicos. La Entrega 4 incorpora recuperación híbrida escalable con Pinecone.

## Entrega 3 — Sistema RAG

El flujo responde preguntas usando únicamente los documentos locales:

```text
Documentos → Chunking → Embeddings → ChromaDB → Retriever → Prompt → LLM → PydanticOutputParser
```

Los archivos de `data/` se fragmentan con `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` en chunks de 500 tokens, con 50 tokens de overlap. Se generan embeddings con `text-embedding-3-small`, se guardan localmente en ChromaDB y se recuperan solo los tres fragmentos más relevantes (`RAG_TOP_K=3`). La respuesta final incluye texto y referencias validadas por Pydantic.

## Pre-entrega 4 — RAG escalable con Pinecone

Esta entrega añade una alternativa cloud al RAG local de la Entrega 3. No reemplaza ChromaDB: indexa el mismo dataset técnico en Pinecone Serverless y combina dos recuperadores antes de devolver el top-5.

```text
Documentos
    ↓
Chunking (600 tokens, overlap 75)
    ↓
OpenAI Embeddings (text-embedding-3-small)
    ↓
Pinecone Serverless + BM25
    ↓
Hybrid Retriever (vector 0.6 + BM25 0.4)
    ↓
Top-5 → Evaluación
```

El script de ingesta crea automáticamente el índice si no existe. Usa dimensión `1536`, métrica `cosine` y el namespace configurado para aislar los documentos. Cada chunk guarda `source`, `page`, `category`, `chunk_id` y `text`; para los archivos `.md` y `.txt`, `page=1` representa el documento de origen completo.

## Estructura del repositorio

```text
Proyect-AI_Engineering/
├── data/
│   ├── api_stack.md
│   ├── database_connections.txt
│   └── monitoring.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── base_client.py
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── manager.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── prompt.py
│   │   └── chain.py
│   ├── rag/
│       ├── __init__.py
│       ├── ingestion.py
│       ├── retriever.py
│       ├── prompt.py
│       └── chain.py
│   └── cloud_rag/
│       ├── __init__.py
│       ├── pinecone_setup.py
│       ├── ingestion.py
│       └── retriever.py
├── evaluation/
│   ├── golden_set.json
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── test_schema.py
│   ├── test_clients.py
│   ├── test_manager.py
│   ├── test_pipeline.py
│   ├── test_rag.py
│   └── test_cloud_rag.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── main.py
```

`vectorstore/` se crea después de la primera ejecución y no se versiona.
Los documentos de `data/` tienen entre 700 y 1000 palabras aproximadamente, por lo que permiten observar chunking real de 500 tokens con 50 tokens de overlap.

## Requisitos

- Python 3.12
- Git opcional
- Una API key de OpenAI para la ejecución real de RAG y embeddings

## Instalación paso a paso — Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración

```powershell
copy .env.example .env
```

Luego edita `.env` sin incluir claves en el código. Para la ejecución RAG con OpenAI, completa al menos `OPENAI_API_KEY`.

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=300
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=vectorstore
RAG_TOP_K=3
PINECONE_API_KEY=
PINECONE_INDEX_NAME=technical-rag
PINECONE_NAMESPACE=technical-docs
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

Si se selecciona `LLM_PROVIDER=anthropic`, completa también `ANTHROPIC_API_KEY` y usa un modelo Anthropic en `LLM_MODEL`. OpenAI sigue siendo necesario para `OpenAIEmbeddings`.

Para la Pre-entrega 4 completa además `PINECONE_API_KEY`. El nombre de índice puede mantenerse en su valor por defecto o personalizarse antes de la primera ingesta.

## Ejecutar

```powershell
python main.py
```

El script inicializa o reutiliza el vectorstore, formula una pregunta conocida, imprime la respuesta con referencias, y luego ejecuta una pregunta trampa que debe responder `No lo sé.`.

## Primera ejecución

La primera ejecución:

- lee los archivos `.txt` y `.md` de `data/`;
- fragmenta los documentos en tokens;
- genera embeddings;
- crea `vectorstore/`;
- persiste la colección de ChromaDB.

## Ejecuciones posteriores

Si existe `vectorstore/chroma.sqlite3`, el sistema reutiliza la colección persistida y no vuelve a indexar los documentos.

## Ingesta y evaluación cloud

La Pre-entrega 4 usa `data/` como corpus. Para crear o reutilizar el índice Serverless de Pinecone e insertar los chunks, ejecuta:

```powershell
python -m src.cloud_rag.ingestion
```

Los IDs son deterministas (`archivo + número de chunk`); una nueva ingesta usa `upsert` y reemplaza el mismo chunk en vez de duplicarlo. La recuperación híbrida combina un retriever vectorial de Pinecone y un retriever léxico BM25 mediante LangChain `EnsembleRetriever`, con pesos 0.6 y 0.4. BM25 se implementa localmente mediante `rank-bm25` y favorece términos técnicos exactos.

Para evaluar las cinco preguntas del Golden Set:

```powershell
python evaluation/evaluate.py
```

`golden_set.json` cubre las categorías API, base de datos y monitoreo. Para cada consulta, `evaluate.py` imprime los documentos top-5 y calcula:

- **Recall@5:** `1` si el documento esperado aparece en top-5; `0` si no aparece.
- **Precision@5:** cantidad de resultados cuyo `source` coincide con el único documento esperado, dividida por `5`.

El reporte final muestra el promedio de ambas métricas sobre las cinco preguntas. La ingesta y la evaluación cloud requieren credenciales válidas de OpenAI y Pinecone; los tests no.

## Pregunta conocida

```text
¿Cuál es el máximo de conexiones del pool de PostgreSQL?
```

La evidencia aparece en `data/database_connections.txt`; la respuesta debe indicar que el máximo es 20 y referenciar ese archivo.

## Pregunta trampa

```text
¿Qué proveedor de pagos utiliza la API?
```

La información no aparece en el dataset. El prompt grounded exige una salida equivalente a:

```json
{
  "respuesta": "No lo sé.",
  "referencias": []
}
```

## Tests

```powershell
python -m pytest
```

Los tests usan mocks y fakes para embeddings, LLM y vectorstore; no consumen crédito ni requieren API keys.

### Troubleshooting de pytest en Windows

Si Windows bloquea una carpeta temporal, ejecuta opcionalmente:

```cmd
set TMP=%CD%\tmp
set TEMP=%CD%\tmp
mkdir tmp
python -m pytest
```

## Variables de entorno

`src/config.py` carga `.env` una sola vez y centraliza estas variables:

| Variable | Uso | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | SDK OpenAI y embeddings | vacío |
| `ANTHROPIC_API_KEY` | SDK Anthropic | vacío |
| `LLM_PROVIDER` | Proveedor del LLM | `openai` |
| `LLM_MODEL` | Modelo de chat | `gpt-4o-mini` en `.env.example` |
| `LLM_TEMPERATURE` | Temperatura del modelo | `0.7` |
| `LLM_MAX_TOKENS` | Máximo de tokens de respuesta | `300` |
| `EMBEDDING_MODEL` | Modelo usado al indexar y buscar | `text-embedding-3-small` |
| `CHROMA_PERSIST_DIR` | Carpeta de ChromaDB | `vectorstore` |
| `RAG_TOP_K` | Fragmentos recuperados | `3` |
| `PINECONE_API_KEY` | Credencial de Pinecone Serverless | vacío |
| `PINECONE_INDEX_NAME` | Nombre del índice cloud | `technical-rag` |
| `PINECONE_NAMESPACE` | Espacio aislado de los documentos cloud | `technical-docs` |
| `PINECONE_CLOUD` | Proveedor serverless de Pinecone | `aws` |
| `PINECONE_REGION` | Región serverless de Pinecone | `us-east-1` |

## Seguridad

`.env`, `.venv`, `vectorstore/`, `tmp/`, `__pycache__/`, archivos `.pyc` y `.pytest_cache/` están ignorados por Git. No se incluyen claves reales en el repositorio.

## Notas sobre costos

Los tests no realizan llamadas externas. En cambio, `python main.py` requiere una clave válida y cuota de OpenAI para generar embeddings y usar el LLM configurado; si se usa Anthropic como LLM, también requiere su clave y cuota. La ingesta y evaluación de Pinecone necesitan también una clave válida de Pinecone.
