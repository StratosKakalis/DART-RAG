import operator
from typing import Annotated, Sequence, Literal, Sequence, Optional
from typing import Dict, Any, Tuple
from typing_extensions import TypedDict, List

# region #### Helper States #####
class StepTaskState(TypedDict):
    type: str
    task: str

class QAAnswerState(TypedDict):
    analysis: str
    answer: str
    rel_citations: List[Optional['Citation']]
    success: str
    # rating: int

class Citation(TypedDict):
    content: str                    # Chunk of paper right after retrieval, or more concise extracted exact phrases after extraction.
    title: str
    paper_id: int
    doi: str
    citation_index: Optional[int]   # The index used within the SciQA answer for inline citations.
# endregion

# region #### State Graphs #####
class PlannerState(TypedDict):
    """
    Represents the state of the central planner-orchestrator.
    """
    original_question: str
    plan: List[str]
    current_step: int
    # memory: List[str] # TODO
    plan_exec_state: Optional['PlanExecState']
    final_answer: QAAnswerState

class RagState(TypedDict):
    """
    Represents the state for a Retrieval-Augmented Generation (RAG) process.
    """
    step_question: str
    citations: List[Citation]  
    extracted_citations: List[Citation]                    # This is the main citation list that is used within the agentic system, the original "citations" are only collected for DeepEval evaluation.
    step_answer: QAAnswerState
    is_sufficient: bool                                     # Whether the current citations are sufficient to answer the question

class PlanExecState(TypedDict):
    """
    Represents the state during the execution of a planned sequence of actions.
    """
    planner_state: PlannerState
    step_questions: Annotated[List[StepTaskState], operator.add]    # List of step questions derived from the plan
    step_rag_states: Annotated[List[RagState], operator.add]        # RAG states for each step
    # TODO: Examine a possible plan summary at this point, that can later be used for memory.
    plan_completed: bool
    stop: bool=False
# endregion
