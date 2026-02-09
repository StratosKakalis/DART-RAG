import re
from typing import Dict, List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.graph.utils.base_models import SciQAAnswerFormat
from src.agents.prompts.prompts import SCI_QA_PROMPT
from src.graph.utils.graph_states import Citation, QAAnswerState

BRACKET_RE = re.compile(r"\[([0-9, ]+)\]")

def create_sci_qa_agent(llm):
    """
    Returns a callable sci_qa_agent(question: str, docs: Dict[str, str]) -> dict
    - docs: mapping of doc_id (usually DOI) -> full doc text or excerpt
    """
    parser = PydanticOutputParser(pydantic_object=SciQAAnswerFormat)
    prompt = ChatPromptTemplate.from_template(SCI_QA_PROMPT)
    chain = prompt | llm.with_structured_output(SciQAAnswerFormat)

    def sci_qa_agent(question: str, citations: List[Citation]) -> QAAnswerState:
        """
        Runs the SciQA agent.

        Args:
            question: the user question.
            docs: list of citations

        Returns:
            a dict matching SciQAAnswerFormat (Pydantic validated).
        """
        # Build numbered citation, based on the Citation structure
        numbered_list_lines = []
        for citation in citations:
            doc_index = citation["citation_index"]
            doc_id = citation["paper_id"]
            doi = citation.get("doi", "No DOI")
            title = citation.get("title", "No Title")
            content = citation.get("extracted_content", "No content available.")
            numbered_list_lines.append(f"[{doc_index}]: Title: {title}, paper id: {doc_id}, doi: {doi}, content: {content}")

        numbered_docs = "\n".join(numbered_list_lines) if numbered_list_lines else "None"

        # 2) Call the structured LLM
        result = chain.invoke({
            "question": question,
            "numbered_docs": numbered_docs,
            "format_instructions": parser.get_format_instructions()
        })

        # result is an instance of SciQAAnswerFormat
        output_obj: SciQAAnswerFormat = result

        out = output_obj.model_dump()
        # out = validate_sci_qa_output()

        return out

    return sci_qa_agent
