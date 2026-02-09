from src.graph.utils.graph_states import PlanExecState, RagState, QAAnswerState

def create_rag_node(rag_graph):
    """
    Wraps a RagGraph (subgraph) so it can be used
    as a node in the PlanExecGraph graph.
    """

    def rag_node(state: PlanExecState) -> dict:
        # Build subgraph initial state
        rag_state: RagState = {
            "step_question": state["step_questions"][state["planner_state"]["current_step"]]["task"],
            "citations": [],
            "extracted_citations": [],
            "step_answer": QAAnswerState(
                answer="",
            )
        }

        # Run subgraph
        final_rag_state = rag_graph.invoke(rag_state)

        # Return entire RagState for this step
        return {
            "step_rag_states": [final_rag_state]
        }

    return rag_node
