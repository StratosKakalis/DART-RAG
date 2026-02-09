from src.graph.utils.graph_states import PlannerState, PlanExecState

def create_plan_executor_node(plan_exec_graph):
    """
    Wraps a PlanExecGraph (subgraph) so it can be used
    as a node in the PlannerState graph.
    """

    def plan_executor_node(state: PlannerState) -> dict:
        # Build subgraph initial state
        plan_exec_state: PlanExecState = {
            "planner_state": state,
            "step_rag_states": [],
            "stop": False
        }

        # Run subgraph
        final_plan_exec_state = plan_exec_graph.invoke(plan_exec_state)

        return {
            "plan_exec_state": final_plan_exec_state
        }

    return plan_executor_node
