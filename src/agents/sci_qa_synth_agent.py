import re
from typing import Dict, List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.graph.utils.base_models import SciQASynthAnswerFormat
from src.agents.prompts.prompts import SCI_QA_SYNTH_PROMPT
from src.graph.utils.graph_states import Citation, QAAnswerState

BRACKET_RE = re.compile(r"\[([0-9, ]+)\]")

def create_sci_qa_synth_agent(llm):
    """
    Returns a callable sci_qa_agent(question: str, docs: Dict[str, str]) -> dict
    - docs: mapping of doc_id (usually DOI) -> full doc text or excerpt
    """
    parser = PydanticOutputParser(pydantic_object=SciQASynthAnswerFormat)
    prompt = ChatPromptTemplate.from_template(SCI_QA_SYNTH_PROMPT)
    chain = prompt | llm.with_structured_output(SciQASynthAnswerFormat)

    def sci_qa_synth_agent(original_question: str, step_rag_states) -> QAAnswerState:
        """
        Runs the SciQA agent.

        Args:
            question: the user question.
            docs: list of citations

        Returns:
            a dict matching SciQASynthAnswerFormat (Pydantic validated).
        """
        # Collect past step questions and answers including citations
        prev_step_questions = []
        prev_step_answers = []
        prev_step_citations = []
        for rag_state in step_rag_states:
            prev_step_questions.append(rag_state["step_question"])
            prev_step_answers.append(rag_state["step_answer"]["answer"])
            prev_step_citations.append(rag_state["step_answer"]["rel_citations"])

        # Build the history
        history = ""
        if prev_step_questions:
            for i, (q, a, c) in enumerate(zip(prev_step_questions, prev_step_answers, prev_step_citations)):
                history += f"Previous step {i + 1}:\n"
                history += f"Question: {q}\n"
                history += f"Answer: {a}\n\n"
                history += "Citations:\n"
                for cite in c:
                    history += f"[{cite['citation_index']}]: Paper: {cite['title']}, paper id: {cite['paper_id']}, doi: {cite['doi']}, content: {cite['content']}\n"
        else:
            history = "None"

        # 2) Call the structured LLM
        result = chain.invoke({
            "original_question": original_question,
            "history": history,
            "format_instructions": parser.get_format_instructions()
        })

        # result is an instance of SciQASynthAnswerFormat
        output_obj: SciQASynthAnswerFormat = result

        out = output_obj.model_dump()

        return out

    return sci_qa_synth_agent
