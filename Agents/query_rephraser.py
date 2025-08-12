import json
from langchain_ibm import ChatWatsonx
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential # Ensure tenacity is installed
from config.logger_config import get_logger, WorkNotesManager
from utils.utils import load_prompts 
from typing import AsyncIterator, Dict, Any, List # Added for type hinting generators

logger = get_logger("query_rephraser")

class QueryRephraser:
    """
    Agent responsible for decomposing complex queries into sub-questions
    and refining queries based on feedback.
    """

    def __init__(self, llm: ChatWatsonx):
        self.llm = llm
        prompts = load_prompts()
        self.query_decomposer_prompt = prompts.get('query_decomposer_prompt', '')
        self.query_refiner_prompt = prompts.get('query_refiner_prompt', '')

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=1, max=4))
    # MODIFIED: Changed return type to AsyncIterator to yield notes and then a final result
    async def decompose_query(self, work_notes_manager: WorkNotesManager, query: str, context: str = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Takes a complex initial query and breaks it into a list of sub-questions.
        This method is now an asynchronous generator that yields work notes and the final result.
        """
        note_content = f"Decomposing complex query: '{query}'"
        work_notes_manager.add_note("QueryDecomposer", note_content)
        yield {"type": "work_note", "content": f"[QueryDecomposer] {note_content}"} # Yield the note
        
        prompt_template = ChatPromptTemplate.from_template(self.query_decomposer_prompt)
        chain = prompt_template | self.llm | JsonOutputParser()
        context_str = context if context else "No context provided."
        
        result_content = {} # Initialize to capture the result

        try:
            # FIX: Use await chain.ainvoke() for asynchronous LLM interaction
            result_content = await chain.ainvoke({"original_question": query, "context": context_str})
            
            if result_content.get("rephrased_question"):
                note_content = f"Observation: Query decomposed into {len(result_content['rephrased_question'])} sub-questions."
                work_notes_manager.add_note("QueryDecomposer", note_content)
                yield {"type": "work_note", "content": f"[QueryDecomposer] {note_content}"}
            
        except Exception as e:
            logger.error(f"Query Decomposer failed: {e}", exc_info=True)
            note_content = f"ERROR: Query Decomposer failed: {str(e)}"
            work_notes_manager.add_note("QueryDecomposer", note_content)
            yield {"type": "work_note", "content": f"[QueryDecomposer] {note_content}"}
            result_content = {"rephrased_question": [query], "thought": "Failed to decompose, using original query."} # Fallback
            
        # Yield the final result for this operation
        yield {"type": "decomposer_result", "content": result_content}

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=1, max=4))
    # MODIFIED: Changed return type to AsyncIterator to yield notes and then a final result
    async def refine_query(self, work_notes_manager: WorkNotesManager, query: str, review: str, context: str = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Takes a failed query and a "review" (the Thought from the Evidence Agent)
        and refines it into a new, targeted question for the missing information.
        This method is now an asynchronous generator that yields work notes and the final result.
        """
        note_content = f"Generating targeted sub-question based on feedback. Feedback: '{review}'"
        work_notes_manager.add_note("QueryRephraser", note_content)
        yield {"type": "work_note", "content": f"[QueryRephraser] {note_content}"} # Yield the note
        
        prompt = ChatPromptTemplate.from_template(self.query_refiner_prompt)
        chain = prompt | self.llm | JsonOutputParser()
        context_str = context if context else "No context provided."

        refined_question = "" # Initialize

        try:
            # FIX: Use await chain.ainvoke() for asynchronous LLM interaction
            result = await chain.ainvoke({
                "original_question": query,
                "review": review,
                "context": context_str
            })
            refined_question = result.get("rephrased_question", "")
        except Exception as e:
            logger.error(f"Query Rephraser failed: {e}", exc_info=True)
            note_content = f"ERROR: Query Rephraser failed: {str(e)}"
            work_notes_manager.add_note("QueryRephraser", note_content)
            yield {"type": "work_note", "content": f"[QueryRephraser] {note_content}"}
            refined_question = query # Fallback to original query
            
        note_content = f"Observation: Generated new targeted query: '{refined_question}'"
        work_notes_manager.add_note("QueryRephraser", note_content)
        yield {"type": "work_note", "content": f"[QueryRephraser] {note_content}"}

        # Yield the final result for this operation
        yield {"type": "rephrased_question", "content": refined_question}