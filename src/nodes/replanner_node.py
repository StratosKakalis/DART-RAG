from src.graph.utils.graph_states import PlanExecState
from src.agents.replanner_agent import create_replanner_agent

def create_replanner_node(llm):
    planner_agent = create_replanner_agent(llm)

    def replan(state: PlanExecState) -> dict:
        question = state["planner_state"]["original_question"]
        plan = state["planner_state"]["plan"]
        
        # Extract answers from the nested structure
        step_rag_states = state["step_rag_states"]
        prev_step_answers = [s["step_answer"]["answer"] for s in step_rag_states]

        formatted_lines = []
        for i, step_text in enumerate(plan):
            step_num = i + 1 # Convert 0-index to 1-based step number
            
            if i < len(prev_step_answers) - 1:
                # Completed steps (We have an answer for it)
                formatted_lines.append(f"\nCompleted Step {step_num}: {step_text} \nAnswer {step_num}: {prev_step_answers[i]}")
            elif i == len(prev_step_answers) - 1:
                # Previous step (indicate to the model that it needs to place emphasis on this step.)
                formatted_lines.append(f"\nCompleted Step {step_num}: {step_text} (this is the previous step that was just executed).\nAnswer {step_num}: {prev_step_answers[i]}")
            else:
                # FUTURE STEPS (just the existing steps )
                formatted_lines.append(f"\nFuture Step {step_num}: {step_text}")

        # Join them all into one block
        plan_execution_context = "\n".join(formatted_lines)

        # Pass the single formatted string to your agent
        results = planner_agent(question, plan_execution_context)

        # Update the plan (PlannerState)
        planner_state = state["planner_state"]
        planner_state["plan"] = results["plan"]
        
        return {
            "planner_state": planner_state,
            "plan_completed": results["plan_completed"]
        }

    return replan