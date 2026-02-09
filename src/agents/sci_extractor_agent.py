from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from typing import Dict, List

from src.graph.utils.base_models import RelevanceAssessmentOutput, SciExtractorOutput
from src.graph.utils.graph_states import Citation
from src.agents.prompts.prompts import SCI_EXTRACTOR_PROMPT

def create_sci_extractor_agent(llm):
    parser = PydanticOutputParser(pydantic_object=SciExtractorOutput)
    prompt = ChatPromptTemplate.from_template(SCI_EXTRACTOR_PROMPT)
    chain = prompt | llm.with_structured_output(SciExtractorOutput)

    def sci_extractor_agent(relevant_docs: List[Citation], question: str) -> List[Citation]:
        """
        Input:
            relevant_docs: List of Typed Dict object with "doc_id", "extracted_content" fields
            question: str
        Output:
            List of Citations following the same structure as input relevant_docs, but with updated "extracted_content" fields.
        """
        output: SciExtractorOutput = chain.invoke({
            "documents": relevant_docs,
            "question": question,
            "format_instructions": parser.get_format_instructions(),
        })
        return output.extracted_content

    return sci_extractor_agent
