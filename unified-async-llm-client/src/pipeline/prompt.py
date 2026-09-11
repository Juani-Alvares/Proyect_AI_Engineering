"""Prompt used by the technical extraction pipeline."""

from langchain_core.prompts import ChatPromptTemplate


technical_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extrae entidades técnicas del texto recibido.
Identifica las tecnologías mencionadas, clasifica el nivel de criticidad como
exactamente baja, media o alta, y redacta un resumen técnico breve. Usa solo
información presente en el texto y devuelve todos los campos solicitados.""",
        ),
        ("human", "Texto técnico a analizar:\n{text}"),
    ]
)
