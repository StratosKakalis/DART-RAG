from src.graph.utils.graph_states import PlanExecState, StepTaskState
from src.agents.query_rewriter_agent import create_query_rewriter_agent

def create_query_rewriter_node(llm):
    query_rewriter_agent = create_query_rewriter_agent(llm)

    def rewrite_step(state: PlanExecState) -> dict:
        """
        LangGraph node.
        Input: RagState
        Output: partial RagState update
        """
        
        # Get current plan step.
        plan_step = state["planner_state"]["plan"][state["planner_state"]["current_step"] + 1]
        
        rewrite_step_out = query_rewriter_agent(
            plan_step=plan_step
        )

        step_task = StepTaskState(
            type=rewrite_step_out["type"],
            task=rewrite_step_out["task"]
        )

        # Update the planner state with the new current step index
        planner_state = state["planner_state"]
        planner_state["current_step"] += 1

        return {
            "planner_state": planner_state,
            "step_questions": [step_task],
        }

    return rewrite_step
  