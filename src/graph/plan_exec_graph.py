from src.nodes.step_definer_node import create_step_definer_node
from src.nodes.query_rewriter_node import create_query_rewriter_node
from src.nodes.replanner_node import create_replanner_node
from src.nodes.rag_wrapper import create_rag_node
from langgraph.graph import StateGraph, START, END

from src.graph.rag_graph import RAGGraph
from src.graph.utils.graph_states import PlanExecState

MAX_RECURSION_DEPTH = 4         # Maximum alowed number of plan steps. To prevent from infinite loops and high cost. 

class PlanExecGraph:
    """
    Execution sub-graph.
    Loops over the plan steps, converts them into tasks,
    runs RAG subgraph for each step.
    """
    def __init__(self, llm, rr: bool = False):
        # Create the nodes
        if not rr:
            self.step_definer_node = create_step_definer_node(llm)
        else: 
            self.replanner_node = create_replanner_node(llm)
            self.query_rewriter_node = create_query_rewriter_node(llm)

        # Compile the rag subgraph and use a wrapper node to add it to this graph.
        self.rag_graph = RAGGraph(llm, rr=rr)
        self.rag_node = create_rag_node(self.rag_graph.graph)

        # Create the graph builder
        self.graph_builder = StateGraph(PlanExecState)
        # If reflective replanning is not used, use step definer, otherwirse step defining is left up the the planner.
        if not rr:
            self.graph_builder.add_node("step_definer", self.step_definer_node)    
        else:
            self.graph_builder.add_node("replanner", self.replanner_node)
            self.graph_builder.add_node("query_rewriter", self.query_rewriter_node)

        self.graph_builder.add_node("rag", self.rag_node)
        # Add edges
        if not rr: 
            self.graph_builder.add_edge(START, "step_definer")
            self.graph_builder.add_edge("step_definer", "rag")
        else:
            self.graph_builder.add_edge(START, "query_rewriter")
            self.graph_builder.add_edge("query_rewriter", "rag")
            self.graph_builder.add_edge("rag", "replanner")
        
        # Conditional edge from RAG
        if not rr:
            self.graph_builder.add_conditional_edges(
                "rag",
                self.should_continue,
                {
                    "continue": "step_definer",
                    "end": END
                }
            )
        else:
            self.graph_builder.add_conditional_edges(
                "replanner", 
                self.define_plan_execution_route,
                {
                    "execute_plan": "query_rewriter",
                    "finalize": END
                }
            )

        self.graph = self.graph_builder.compile()

    @staticmethod
    def define_plan_execution_route(state: PlanExecState) -> str:
        plan_completed = state.get("plan_completed", False)
        current_step = state.get("planner_state").get("current_step")

        if plan_completed or current_step>=MAX_RECURSION_DEPTH:
            return "finalize"
        return "execute_plan"

    @staticmethod
    def should_continue(state: PlanExecState) -> str:
        current_step = state["planner_state"]["current_step"]
        total_steps = len(state["planner_state"]["plan"])

        if current_step < total_steps - 1:
            return "continue"
        return "end"