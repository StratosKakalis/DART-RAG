from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.graph.utils.base_models import QueryRewriterOutput
from src.graph.utils.graph_states import StepTaskState

from src.agents.prompts.prompts import QUERY_REWRITER_PROMPT

def create_query_rewriter_agent(llm):
    """
    Creates a node that converts a plan step (string) 
    into a StepTaskState (task type + concrete task).
    """

    parser = PydanticOutputParser(pydantic_object=QueryRewriterOutput)
    prompt = ChatPromptTemplate.from_template(QUERY_REWRITER_PROMPT)

    chain = prompt | llm.with_structured_output(QueryRewriterOutput)

    def query_rewriter_agent(plan_step) -> StepTaskState:
        """
        rewrites the plan step into a task.
        """
        
        output: QueryRewriterOutput = chain.invoke(
            {
                "plan_step": plan_step,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        return {
            "type": "question-answering",     
            "task": output.question,
        }

    return query_rewriter_agent
