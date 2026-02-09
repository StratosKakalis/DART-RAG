from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.graph.utils.base_models import RelevanceAssessmentOutput, SciExtractorOutput
from src.agents.prompts.prompts import RELEVANCE_PROMPT


def create_relevance_agent(llm):
    parser = PydanticOutputParser(pydantic_object=RelevanceAssessmentOutput)
    prompt = ChatPromptTemplate.from_template(RELEVANCE_PROMPT)
    chain = prompt | llm.with_structured_output(RelevanceAssessmentOutput)

    def relevance_agent(documents: dict, question: str) -> dict:
        """
        Input:
            documents: dict of {doc_id: doc_text}
            question: str
        Output:
            dict of relevant documents {doc_id: doc_text}
        """
        output: RelevanceAssessmentOutput = chain.invoke({
            "documents": documents,
            "question": question,
            "format_instructions": parser.get_format_instructions(),
        })
        return output.relevant_docs

    return relevance_agent