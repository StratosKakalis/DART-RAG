from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.graph.utils.graph_states import PlannerState
from src.graph.utils.base_models import RePlannerOutput
from src.agents.prompts.prompts import REPLANNER_PROMPT


def create_replanner_agent(llm):
    """
    Creates a RePlanner (Reflective Planner) node that generates a plan based on the original question and the previous steps.
    """

    parser = PydanticOutputParser(pydantic_object=RePlannerOutput)

    prompt = ChatPromptTemplate.from_template(REPLANNER_PROMPT)

    # Build the actual runnable chain
    chain = prompt | llm.with_structured_output(RePlannerOutput)

    def replanner_agent(question, plan_exec_context) -> PlannerState:
        """
        RePlanner node reads: 
        Original Question.
        Existing Plan.
        Current step.
        Previous Step Answers.
        Produces: state["plan"]
        Leaves: final_answer unchanged
        """
        
        result: RePlannerOutput = chain.invoke(
            {
                "question": question,
                "plan_execution_context": plan_exec_context,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # Return updated state
        return {
            "plan": result.plan,
            "plan_completed": result.plan_completed
        }

    return replanner_agent
