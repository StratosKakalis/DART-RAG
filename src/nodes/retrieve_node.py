from src.graph.utils.graph_states import RagState, Citation
from src.tools.rag_tool import RAGClient
import os
from dotenv import load_dotenv

load_dotenv()
RAG_ENCPOINT_URL = os.getenv("RAG_ENDPOINT_URL")
RAG_API_KEY = os.getenv("RAG_API_KEY")

def create_retrieve_node():

    def retrieve(state: RagState) -> dict:
        """
        LangGraph node.
        Input: RagState
        Output: partial RagState update
        """
        rag_client = RAGClient(endpoint_url=RAG_ENCPOINT_URL, api_key=RAG_API_KEY)

        # Perform RAG on the question and get relevant paper chunks.
        step_task = state["step_question"]
        results = rag_client.retrieve(question=step_task, k=20)

        # Keep only the first 5 results:
        results = results[:5]

        # Extract the chunk text, title, paper id and doi from each result
        retrieved_docs: list[Citation] = []
        for res in results:
            chunk_text = res.get("chunk_text", "")
            title = res.get("title", "")
            paper_id = res.get("paper_id", "")
            doi = res.get("doi", "")
            citation = Citation(
                content=chunk_text,
                title=title,
                paper_id=paper_id,
                doi=doi
            )
            retrieved_docs.append(citation)

        return {
            "citations": retrieved_docs,
            "extracted_citations": retrieved_docs
        }

    return retrieve
  