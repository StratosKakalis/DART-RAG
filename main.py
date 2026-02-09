from dotenv import load_dotenv
import os

from src.graph.planner_graph import PlannerGraph
from src.graph.utils.graph_states import PlannerState
from agentic_rag import DART_RAG

load_dotenv()

if __name__ == "__main__":
    rag_system = DART_RAG()
    question = "What are the key benefits of using a RAG architecture in AI applications?"
    answer = rag_system.answer_question(question)
    print("Final Answer:", answer)