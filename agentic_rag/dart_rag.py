import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

from src.graph.planner_graph import PlannerGraph
from src.graph.utils.graph_states import PlannerState


class DART_RAG:
    def __init__(self, rr=False):
        # Initialize LLM
        provider_preferences = {
            "order": [
                "Phala", 
                "DeepInfra", 
                "Novita", 
                "SiliconFlow", 
                "DeepSeek"
            ],
            "allow_fallbacks": True
        }
        llm = ChatOpenAI(
            model=os.environ["AGENT_MODEL_NAME"],
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_API_BASE"],
            temperature=0.2,       
            max_tokens=8192,
            # streaming=True,        # Enable streaming to capture TTFT
            # stream_usage=True,
            extra_body = {
                "provider": provider_preferences,
                # "repetition_penalty": 1.1,                      # Add a repetition penalty because some of the smaller models sometimes get stuck in infinite loops.
            },
        )

        # Build planner graph
        self.planner_graph = PlannerGraph(llm, rr)

    def answer_question(self, question: str, config) -> str:
        # Initialize graph state (TypedDict = plain dict)
        planner_state: PlannerState = {
            "original_question": question,
            "plan": [],
            "current_step": -1,
            "final_answer": "",
        }

        # Invoke the compiled graph
        final_state: PlannerState = self.planner_graph.graph.invoke(
            planner_state, 
            config=config
        )

        return final_state
