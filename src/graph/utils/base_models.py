from typing_extensions import List, Dict
from pydantic import BaseModel, Field
from src.graph.utils.graph_states import Citation

#### Base Models for LLM Output Formatting. #####
class PlannerOutput(BaseModel):
    analysis: str = Field(
        description="Your step-by-step reasoning about how to solve the question."
    )
    plan: List[str] = Field(
        description="A sorted list of steps/subtasks that should be executed to answer the question."
    )

class RePlannerOutput(BaseModel):
    analysis: str = Field(
        description="Your step-by-step reasoning about the system's next steps."
    )
    plan: List[str] = Field(
        description="A sorted list of steps that should be executed to answer the question."
    )
    plan_completed: bool = Field(description="Boolean output (True or False), indicate if there are no remaining steps to execute.")

class StepDefinerOutput(BaseModel):
    sub_question: str = Field(
        description="A clear and concise sub-question that can be answered independently."
    )
    task_type: str = Field(
        description="The type of the task."
    ) 

class QueryRewriterOutput(BaseModel):
    question: str = Field(
        description="A clear and concise question that captures the essence of the plan step."
    )

class RelevanceAssessmentOutput(BaseModel):
    relevant_docs: List[str] = Field(
        description="List of paper id's of relevant documents. Includes only paper segments that are crucial for answering the question."
    )

class SufficiencyAssessmentOutput(BaseModel):
    analysis: str = Field(description="Your thoughts, analysis about the question and the context. Think step-by-step")
    sufficient: bool = Field(description="Boolean output (True or False), indicate if the provided documents are sufficient to answer the question.")

class SciExtractorOutput(BaseModel):
    extracted_content: Dict[str, str] = Field(
        description="Dictionary where keys are paper IDs and values are the exact relevant phrases for answering the question."
    )

class SciQAAnswerFormat(BaseModel):
    analysis: str = Field(description="Your thoughts, analysis about the question and the context. Think step-by-step")
    answer: str = Field(description="The answer for the question")
    rel_citation_ids: List[str] = Field(description="List of paper id's in the order that they appear cited in the final answer.")
    # rel_citations: List[Citation] = Field(description="List of citations used to answer the question.")
    success: bool = Field(description="Boolean output (True or False), indicate if you can answer or not")
    # rating: int = Field( default=None, description="How confident, from 0 to 10. The more evidence and agreement, the more confident")

class SciQASynthAnswerFormat(BaseModel):
    analysis: str = Field(description="Your thoughts, analysis about the question and the context. Think step-by-step")
    answer: str = Field(description="The answer for the question")
    rel_citation_ids: List[str] = Field(description="List of paper id's in the order that they appear cited in the final answer.")
    # rel_citations: List[Citation] = Field(description="List of citations used to answer the question. Make sure that each citation's 'citation_index' corresponds to the **global** index used in the answer.")
    success: bool = Field(description="Boolean output (True or False), indicate if you can answer or not")
    # rating: int = Field( default=None, description="How confident, from 0 to 10. The more evidence and agreement, the more confident")
