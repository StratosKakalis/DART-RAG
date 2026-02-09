from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.graph.utils.base_models import SufficiencyAssessmentOutput, SciExtractorOutput
from src.agents.prompts.prompts import SUFFICIENCY_PROMPT


def create_sufficiency_agent(llm):
    parser = PydanticOutputParser(pydantic_object=SufficiencyAssessmentOutput)
    prompt = ChatPromptTemplate.from_template(SUFFICIENCY_PROMPT)
    chain = prompt | llm.with_structured_output(SufficiencyAssessmentOutput)

    def sufficiency_agent(documents: dict, question: str) -> dict:
        """
        Input:
            documents: dict of {doc_id: doc_text}
            question: str
        Output:
            dict of relevant documents {doc_id: doc_text}
        """
        output: SufficiencyAssessmentOutput = chain.invoke({
            "documents": documents,
            "question": question,
            "format_instructions": parser.get_format_instructions(),
        })
        return output

    return sufficiency_agent