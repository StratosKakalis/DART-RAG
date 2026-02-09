from src.graph.utils.graph_states import PlanExecState, StepTaskState
from src.agents.step_definer_agent import create_step_definer_agent

def create_step_definer_node(llm):
    step_definer_agent = create_step_definer_agent(llm)

    def define_step(state: PlanExecState) -> dict:
        """
        LangGraph node.
        Input: RagState
        Output: partial RagState update
        """
        # Get original question.
        original_question = state["planner_state"]["original_question"]
        # Get current plan step.
        plan_step = state["planner_state"]["plan"][state["planner_state"]["current_step"] + 1]
        # Get previous step questions and asnwers.
        prev_step_questions = []
        prev_step_answers = []
        for rag_state in state["step_rag_states"]:
            prev_step_questions.append(rag_state["step_question"])
            prev_step_answers.append(rag_state["step_answer"]["answer"])

        step_definition = step_definer_agent(
            original_question=original_question,
            plan_step=plan_step, 
            prev_step_questions=prev_step_questions,
            prev_step_answers=prev_step_answers
        )

        step_task = StepTaskState(
            type=step_definition["type"],
            task=step_definition["task"]
        )

        # Update the planner state with the new current step index
        planner_state = state["planner_state"]
        planner_state["current_step"] += 1

        return {
            "planner_state": planner_state,
            "step_questions": [step_task],
        }

    return define_step
  