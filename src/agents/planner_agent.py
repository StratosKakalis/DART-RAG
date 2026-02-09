from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.graph.utils.graph_states import PlannerState
from src.graph.utils.base_models import PlannerOutput
from src.agents.prompts.prompts import PLANNER_PROMPT


def create_planner_agent(llm):
    """
    Creates a planner node that generates a plan based on the original question.
    """

    parser = PydanticOutputParser(pydantic_object=PlannerOutput)

    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

    # Build the actual runnable chain
    chain = prompt | llm.with_structured_output(PlannerOutput)

    def planner_agent(question) -> PlannerState:
        """
        Planner node reads: state["original_question"]
        Produces: state["plan"]
        Leaves: final_answer unchanged
        """
        
        result: PlannerOutput = chain.invoke(
            {
                "question": question,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # Return updated state
        return {
            "plan": result.plan
        }

    return planner_agent
