"""Grounded prompt and output parser for RAG responses."""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..schemas import RAGResponse

rag_output_parser = PydanticOutputParser(pydantic_object=RAGResponse)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Responde únicamente usando el CONTEXTO recuperado. No inventes
información ni uses conocimiento externo. Si el contexto no contiene evidencia
suficiente, responde exactamente "No lo sé." y usa una lista vacía de referencias.
Cuando exista evidencia, incluye solo los nombres de archivos fuente usados.

{format_instructions}""",
        ),
        (
            "human",
            "PREGUNTA:\n{question}\n\nCONTEXTO:\n{context}",
        ),
    ]
).partial(format_instructions=rag_output_parser.get_format_instructions())
