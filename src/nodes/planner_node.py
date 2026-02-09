from src.graph.utils.graph_states import PlannerState, Citation
from src.agents.planner_agent import create_planner_agent

def create_planner_node(llm):
    planner_agent = create_planner_agent(llm)

    def plan(state: PlannerState) -> dict:
        """
        LangGraph node.
        Input: PlannerState
        Output: partial PlannerState update
        """
        question = state["original_question"]
        
        results = planner_agent(question)
        # Extract "Plan" from results
        plan = results["plan"]

        return {
            "plan": plan
        }

    return plan