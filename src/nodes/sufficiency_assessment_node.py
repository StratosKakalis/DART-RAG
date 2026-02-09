from src.graph.utils.graph_states import RagState, Citation, QAAnswerState
from src.agents.sufficiency_assessment_agent import create_sufficiency_agent

def create_sufficiency_node(llm):
    suff_assesment_agent = create_sufficiency_agent(llm)

    def extract(state: RagState) -> dict:
        """
        LangGraph node.
        Input: RagState
        Output: partial RagState update
        """
        question = state["step_question"]
        citations: list[Citation] = state["extracted_citations"]

        if not citations:
            return {"extracted_citations": []}

        relevant_citations = suff_assesment_agent(
            documents=citations,
            question=question,
        )

        analysis = relevant_citations.analysis
        is_sufficient = relevant_citations.sufficient
        
        # If answer is sufficient then just return the appropriate bool.
        if is_sufficient:
            return {
                "is_sufficient": is_sufficient
            }
        else:
        # Else if data is not sufficient, return the analysis as the step answer and interrupt the execution of the step.
            answer_state = QAAnswerState(
                analysis="",
                answer=analysis,
                rel_citations=[],
                success="",
                rating=0
            )
            return {
                "step_answer": answer_state,
                "is_sufficient": is_sufficient
            }

    return extract
