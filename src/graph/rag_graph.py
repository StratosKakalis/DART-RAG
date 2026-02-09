from langgraph.graph import StateGraph, START, END

from src.nodes.retrieve_node import create_retrieve_node
from src.nodes.rel_assesment_node import create_relevance_node
from src.nodes.extract_node import create_extract_node
from src.nodes.sufficiency_assessment_node import create_sufficiency_node
from src.nodes.sqa_node import create_sqa_node
from src.graph.utils.graph_states import PlanExecState, StepTaskState, RagState


def route_sufficiency(state: RagState):
    """
    Determines the next node based on the sufficiency assessment.
    """
    is_sufficient = state.get("is_sufficient", False)     

    if is_sufficient:
        return "sci_qa"  # Go to answer generation
    else:
        return "end"

class RAGGraph:
    """
    Step execution RAG sub-graph.
    Performs retrieval, extraction and answer generation for a given step question.
    """
    def __init__(self, llm, rr: bool = False):
        # Create all nodes
        self.retrieve_node = create_retrieve_node()
        self.rel_assess_node = create_relevance_node(llm)
        self.extract_node = create_extract_node(llm)
        if rr:
            self.suff_assess_node = create_sufficiency_node(llm)
        self.sqa_node = create_sqa_node(llm)

        # Create the graph builder
        self.builder = StateGraph(RagState)

        # Register nodes
        self.builder.add_node("retrieve", self.retrieve_node)
        self.builder.add_node("rel_assess", self.rel_assess_node)
        self.builder.add_node("extract", self.extract_node)
        # Sufficiency assessment node (only for reflective replanning)
        if rr:
            self.builder.add_node("suff_assess", self.suff_assess_node)
        self.builder.add_node("sci_qa", self.sqa_node)

        # Define edges
        self.builder.add_edge(START, "retrieve")
        self.builder.add_edge("retrieve", "rel_assess")
        self.builder.add_edge("rel_assess", "extract")
        
        # Non reflective:
        if not rr:
            self.builder.add_edge("extract", "sci_qa")
        else:
            # Sufficiency Assessment
            # self.builder.add_edge("extract", "suff_assess")
            # self.builder.add_conditional_edges("suff_assess", route_sufficiency,
            #     {
            #         "sci_qa": "sci_qa",
            #         "end": END
            #     }
            # )
            
            # Without sufficiency assessment
            self.builder.add_edge("extract", "sci_qa")

        self.builder.add_edge("sci_qa", END)

        # Compile graph
        self.graph = self.builder.compile()