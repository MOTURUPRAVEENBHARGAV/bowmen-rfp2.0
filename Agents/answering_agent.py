import json
from typing import Dict, Any, AsyncIterator
from langchain_ibm import ChatWatsonx
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential

from config.logger_config import get_logger, WorkNotesManager
from utils.utils import load_prompts

logger = get_logger("answering_agent")

class AnswerAgent:
    """
    Generates and analyzes the final answer based on collected evidence.
    This agent uses robust prompts that return structured JSON, simplifying parsing.
    """

    def __init__(self, llm: ChatWatsonx):
        self.llm = llm
        prompts = load_prompts()
        self.answer_generator_prompt = prompts.get('answer_generator_prompt', '')
        self.answer_analyzer_prompt = prompts.get('answer_analyzer_prompt', '')

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def make_final_answer(self, work_notes_manager: WorkNotesManager, query: str, evidence: str, context: str = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Generates the final answer from evidence using a prompt that returns clean JSON.
        This method yields work notes and the final answer content.
        """
        note_content = "Generating final answer for query."
        logger.info(note_content)
        work_notes_manager.add_note("Answer_Agent", note_content)
        yield {"type": "work_note", "content": f"[Answer_Agent] {note_content}"}

        prompt_template = ChatPromptTemplate.from_template(self.answer_generator_prompt)
        # The prompt is designed to return a JSON object, so we use JsonOutputParser directly.
        chain = prompt_template | self.llm | JsonOutputParser()
        context_str = context if context else "No context provided."
        final_answer_text = ""

        try:
            # Ainvoke the chain and expect a dictionary from the JsonOutputParser.
            response_dict = await chain.ainvoke({"query": query, "evidence": evidence, "context": context_str})
            
            # Directly access the 'final_answer' key as defined in our robust prompt.
            final_answer_text = response_dict.get('final_answer', "Could not extract the answer from the model's response.")
            
            note_content = "Final answer generated successfully."
            work_notes_manager.add_note("Answer_Agent", note_content)
            yield {"type": "work_note", "content": f"[Answer_Agent] {note_content}"}

        except Exception as e:
            logger.error(f"Error generating final answer: {e}", exc_info=True)
            note_content = f"ERROR: Failed to generate or parse final answer. Error: {str(e)}"
            work_notes_manager.add_note("Answer_Agent", note_content)
            yield {"type": "work_note", "content": f"[Answer_Agent] {note_content}"}
            final_answer_text = "I could not generate a comprehensive answer at this time due to an internal error."

        # Yield the final answer as a distinct event
        yield {"type": "final_answer_content", "content": {"final_answer": str(final_answer_text)}}

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def analyze_final_answer(self, work_notes_manager: WorkNotesManager, query: str, answer: str, context: str = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Analyzes the generated answer for completeness, yielding notes and the analysis result.
        """
        note_content = "Analyzing sufficiency of final answer."
        logger.info(note_content)
        work_notes_manager.add_note("Answer_Agent", note_content)
        yield {"type": "work_note", "content": f"[Answer_Agent] {note_content}"}
        
        prompt_template = ChatPromptTemplate.from_template(self.answer_analyzer_prompt)
        chain = prompt_template | self.llm | JsonOutputParser() # This prompt also expects JSON
        context_str = context if context else "No context provided."
        analysis_result = {}

        try:
            response = await chain.ainvoke({"query": query, "answer": answer, "context": context_str})
            analysis_result = response # JsonOutputParser already gives a dict
            
            note_content = f"Answer sufficiency: {analysis_result.get('Sufficient', 'N/A')}, Thought: {analysis_result.get('Thought', 'N/A')}"
            work_notes_manager.add_note("Answer_Agent", note_content)
            yield {"type": "work_note", "content": f"[Answer_Agent] {note_content}"}

        except Exception as e:
            logger.error(f"Error analyzing final answer: {e}", exc_info=True)
            note_content = f"ERROR: Failed to analyze final answer. Error: {str(e)}"
            work_notes_manager.add_note("Answer_Agent", note_content)
            yield {"type": "work_note", "content": f"[Answer_Agent] {note_content}"}
            analysis_result = {"Sufficient": "false", "Thought": "An internal error occurred during answer analysis."}

        # Yield the final analysis result
        yield {"type": "answer_analysis_result", "content": analysis_result}