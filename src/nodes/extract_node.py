from src.graph.utils.graph_states import RagState, Citation
from src.agents.sci_extractor_agent import create_sci_extractor_agent

def create_extract_node(llm):
    sci_extractor_agent = create_sci_extractor_agent(llm)

    def extract(state: RagState) -> dict:
        """
        LangGraph node.
        Input: RagState
        Output: partial RagState update
        """
        question = state["step_question"]

        # These should already be filtered by relevance
        relevant_docs: list[Citation] = state["extracted_citations"]

        if not relevant_docs:
            return {"extracted_citations": []}

        extracted_citations = sci_extractor_agent(
            relevant_docs=relevant_docs,
            question=question
        )

        # Extracted citations is a Dict where the key is the paper id and the value is the extracted content
        # We will change the original citations to include the new extracted content
        extracted_citations_list = []
        for citation in relevant_docs:
            paper_id = citation["paper_id"]
            if paper_id in extracted_citations:
                extracted_content = extracted_citations[paper_id]
                new_citation = citation.copy()
                new_citation["extracted_content"] = extracted_content
                extracted_citations_list.append(new_citation)

        return {
            "extracted_citations": extracted_citations_list
        }

    return extract
  