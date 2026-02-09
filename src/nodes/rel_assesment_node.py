from src.graph.utils.graph_states import RagState, Citation
from src.agents.rel_assesment_agent import create_relevance_agent

def create_relevance_node(llm):
    rel_assessment_agent = create_relevance_agent(llm)

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

        relevant_citations = rel_assessment_agent(
            documents=citations,
            question=question,
        )

        # Relevant citations is a list of paper ids, so filter the original citations
        all_citations = state["extracted_citations"]
        extracted_citations = [citation for citation in all_citations if citation["paper_id"] in relevant_citations]

        # Populate the "citation_index" for the extracted_citations, serially:
        for index, citation in enumerate(extracted_citations):
            citation["citation_index"] = index + 1

        return {
            "extracted_citations": extracted_citations
        }

    return extract
