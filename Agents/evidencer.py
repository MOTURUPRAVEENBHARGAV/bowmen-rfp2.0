import json
import warnings
from langchain_ibm import ChatWatsonx
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
from config.logger_config import get_logger, WorkNotesManager
from utils.utils import load_prompts 
from typing import AsyncIterator, Dict, Any # Added for type hinting generators

warnings.filterwarnings("ignore")
logger = get_logger("evidence_agent")

class Evidence_Agent:
    """
    Implements the final, correct two-stage evidence evaluation process.
    - Stage 1 uses a robust JsonOutputParser for a simple relevance check.
    - Stage 2 performs the detailed Gap Analysis.
    """
    
    def __init__(self, llm: ChatWatsonx):
        self.llm = llm
        prompts = load_prompts()
        self.quick_relevance_prompt = prompts.get('quick_relevance_prompt', '')
        self.gap_analysis_prompt = prompts.get('gap_analysis_prompt', '')

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=1, max=4))
    # MODIFIED: Changed return type to AsyncIterator to yield notes and then a final result
    async def run_quick_relevance_check(self, work_notes_manager: WorkNotesManager, query: str, evidence: str, context: str = None) -> AsyncIterator[Dict[str, Any]]:
        """
        STAGE 1: Rapidly filters a single piece of evidence using a robust JSON check.
        This method is now an asynchronous generator that yields work notes and the final result.
        """
        note_content = f"Performing quick relevance check on evidence: '{evidence[:150]}...'"
        work_notes_manager.add_note("EvidenceFilter", note_content)
        yield {"type": "work_note", "content": f"[EvidenceFilter] {note_content}"} # Yield the note
        
        prompt = ChatPromptTemplate.from_template(self.quick_relevance_prompt)
        chain = prompt | self.llm | JsonOutputParser()
        context_str = context if context else "No context provided."
        
        is_relevant = False # Initialize result
        try:
            # FIX: Use await chain.ainvoke() for asynchronous LLM interaction
            result = await chain.ainvoke({"query": query, "evidence": evidence or "No evidence provided","context": context_str})
            # Safely get the boolean value from the parsed JSON dictionary
            is_relevant = result.get("is_relevant", False)
            note_content = f"Observation: Evidence relevance is {is_relevant}"
            work_notes_manager.add_note("EvidenceFilter", note_content)
            yield {"type": "work_note", "content": f"[EvidenceFilter] {note_content}"}
        except Exception as e:
            logger.error(f"Quick filter agent failed: {e}", exc_info=True)
            note_content = f"ERROR: Could not parse relevance check. Defaulting to false. Error: {str(e)}"
            work_notes_manager.add_note("EvidenceFilter", note_content)
            yield {"type": "work_note", "content": f"[EvidenceFilter] {note_content}"}
            is_relevant = False # Ensure it's false on error

        # Yield the final result for this operation
        yield {"type": "relevance_check_result", "content": is_relevant}


    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=1, max=4))
    # MODIFIED: Changed return type to AsyncIterator to yield notes and then a final result
    async def run_gap_analysis(self, work_notes_manager: WorkNotesManager, query: str, combined_evidence: str, context: str = None) -> AsyncIterator[Dict[str, Any]]:
        """
        STAGE 2: Performs a holistic gap analysis on all collected evidence.
        This method is now an asynchronous generator that yields work notes and the final result.
        """
        note_content = f"Performing gap analysis on combined evidence: '{combined_evidence[:200]}...'"
        work_notes_manager.add_note("EvidenceAnalyzer", note_content)
        yield {"type": "work_note", "content": f"[EvidenceAnalyzer] {note_content}"} # Yield the note
        
        prompt = ChatPromptTemplate.from_template(self.gap_analysis_prompt)
        chain = prompt | self.llm | JsonOutputParser()
        context_str = context if context else "No context provided."
        
        result_content = {} # Initialize result
        try:
            # FIX: Use await chain.ainvoke() for asynchronous LLM interaction
            result_content = await chain.ainvoke({"query": query, "evidence": combined_evidence, "context": context_str})
            note_content = f"Thought: {result_content.get('Thought', 'No reason provided.')}"
            work_notes_manager.add_note("EvidenceAnalyzer", note_content)
            yield {"type": "work_note", "content": f"[EvidenceAnalyzer] {note_content}"}
        except Exception as e:
            logger.error(f"Gap analysis agent failed: {e}", exc_info=True)
            note_content = f"ERROR: Gap analysis failed. Error: {str(e)}"
            work_notes_manager.add_note("EvidenceAnalyzer", note_content)
            yield {"type": "work_note", "content": f"[EvidenceAnalyzer] {note_content}"}
            result_content = {"Sufficient": "false", "Thought": "An internal error occurred during the final evidence analysis."}

        # Yield the final result for this operation
        yield {"type": "gap_analysis_result", "content": result_content}