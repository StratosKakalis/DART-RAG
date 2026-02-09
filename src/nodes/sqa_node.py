from src.graph.utils.graph_states import RagState, Citation, QAAnswerState
from src.agents.sci_qa_agent import create_sci_qa_agent

def create_sqa_node(llm):
    sci_qa_agent = create_sci_qa_agent(llm)

    def sci_qa(state: RagState) -> dict:
        """
        LangGraph node.
        Input: RagState
        Output: partial RagState update
        """
        step_question = state["step_question"]
        citations: list[Citation] = state["extracted_citations"]

        if not citations:
            return {
                "step_answer": QAAnswerState(
                    analysis="",
                    answer="No relevant data found in existing papers. Cannot answer this sub-question.",
                    rel_citations=[],
                    success="Failed",
                    # rating=0
                )
            }

        sci_qa_response = sci_qa_agent(
            question=step_question,
            citations=citations
        )

        # Construct the final QAAnswerState (extract the relevant citation list that the model generates and get all the mentioned citations in the proper format, in a list).
        extracted_citations = state["extracted_citations"]
        rel_citations = [citation for citation in extracted_citations if citation["paper_id"] in sci_qa_response["rel_citation_ids"]]

        return {
            "step_answer": QAAnswerState(
                analysis=sci_qa_response["analysis"],
                answer=sci_qa_response['answer'],
                rel_citations=rel_citations,
                success=sci_qa_response["success"],
            )
        }

    return sci_qa

