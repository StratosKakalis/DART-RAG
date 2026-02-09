from src.nodes.planner_node import create_planner_node
from src.nodes.plan_executor_wrapper import create_plan_executor_node
from src.nodes.final_sqa_node import create_sqas_node
from src.graph.utils.graph_states import PlannerState
from src.graph.plan_exec_graph import PlanExecGraph

from langgraph.graph import StateGraph, START, END

class PlannerGraph:
    """
    Top-level planner graph.
    Runs the planner agent and produces a plan list.
    """

    def __init__(self, llm, rr: bool = False):
        print(f"Reflective Replanning is: {rr}")
        # Create the nodes
        self.planner_node = create_planner_node(llm)
        
        # Compile the plan executor subgraph and use a wrapper node to add it to this graph.
        self.plan_exec_graph = PlanExecGraph(llm, rr=rr) 
        self.plan_executor_node = create_plan_executor_node(self.plan_exec_graph.graph)
        self.final_answer_sqa_node = create_sqas_node(llm)

        # Create the graph builder
        self.graph_builder = StateGraph(PlannerState)
        self.graph_builder.add_node("planner_node", self.planner_node)
        self.graph_builder.add_node("plan_executor_node", self.plan_executor_node)
        self.graph_builder.add_node("final_answer_sqa_node", self.final_answer_sqa_node)

        self.graph_builder.add_edge(START, "planner_node")
        self.graph_builder.add_edge("planner_node", "plan_executor_node")
        self.graph_builder.add_edge("plan_executor_node", "final_answer_sqa_node")
        self.graph_builder.add_edge("final_answer_sqa_node", END)
        
        self.graph = self.graph_builder.compile()
        