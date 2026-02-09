from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.graph.utils.base_models import StepDefinerOutput
from src.graph.utils.graph_states import StepTaskState

from src.agents.prompts.prompts import STEP_DEFINER_PROMPT

def create_step_definer_agent(llm):
    """
    Creates a node that converts a plan step (string) 
    into a StepTaskState (task type + concrete task).
    """

    parser = PydanticOutputParser(pydantic_object=StepDefinerOutput)
    prompt = ChatPromptTemplate.from_template(STEP_DEFINER_PROMPT)

    chain = prompt | llm.with_structured_output(StepDefinerOutput)

    def step_definer_agent(original_question, plan_step, prev_step_questions, prev_step_answers) -> StepTaskState:
        """
        Produces one executable step task.
        """
        
        # Build execution context for the LLM
        history = ""
        if prev_step_questions:
            for i, (q, a) in enumerate(zip(prev_step_questions, prev_step_answers)):
                history += f"Previous step {i + 1}:\n"
                history += f"Question: {q}\n"
                history += f"Answer: {a}\n\n"
        else:
            history = "None"

        output: StepDefinerOutput = chain.invoke(
            {
                "step": plan_step,
                "original_question": original_question,
                "previous_steps": history,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        return {
            "type": output.task_type,     
            "task": output.sub_question,
        }

    return step_definer_agent
