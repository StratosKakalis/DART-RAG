from src.graph.utils.graph_states import PlannerState, Citation, QAAnswerState
from src.agents.sci_qa_synth_agent import create_sci_qa_synth_agent

def create_sqas_node(llm):
    sci_qas_agent = create_sci_qa_synth_agent(llm)

    def sci_qa_synth(state: PlannerState) -> dict:
        """
        LangGraph node.
        Input: RagState
        Output: partial RagState update
        """
        original_question = state["original_question"]
        plan_exec_state = state["plan_exec_state"]
        step_rag_states = plan_exec_state["step_rag_states"]

        if not step_rag_states:
            return {"final_answer": "No steps generated, synthesis failed."}

        sci_qa_response = sci_qas_agent(
            original_question=original_question,
            step_rag_states=step_rag_states
        )

        # Construct the final QAAnswerState (extract the relevant citation list that the model generates and get all the mentioned citations in the proper format, in a list).
        plan_exec_state = state["plan_exec_state"]

        all_extracted_citations = []
        for item in plan_exec_state["step_rag_states"]:
            # If it's a list of of rag states...
            if isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, dict):
                        all_extracted_citations.extend(sub_item.get("extracted_citations", []))
            elif isinstance(item, dict):
                # In the first iteration there is no list, just the previous rag state...
                all_extracted_citations.extend(item.get("extracted_citations", []))

        print(f"All extracted citations are: {all_extracted_citations}")

        rel_citations = [citation for citation in all_extracted_citations if citation["paper_id"] in sci_qa_response["rel_citation_ids"]]

        return {
            "final_answer": QAAnswerState(
                analysis=sci_qa_response["analysis"],
                answer=sci_qa_response['answer'],
                rel_citations=rel_citations,
                success=sci_qa_response["success"],
            )
        }
    
    return sci_qa_synth
  