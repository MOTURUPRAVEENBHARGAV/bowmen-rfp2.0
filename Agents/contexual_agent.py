import json
import httpx
from typing import Dict, Any, List, AsyncIterator # Added AsyncIterator
from langchain_ibm import ChatWatsonx
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.logger_config import get_logger, WorkNotesManager
from utils.utils import load_prompts 

logger = get_logger("contextual_agent")

class ContextualAgent:
    """
    Manages all context: finds relevant documents, decides if context is needed,
    validates relevance, and generates final advisory strings.
    """

    def __init__(self, llm: ChatWatsonx, collection_name: str,api_url :str):
        self.llm = llm
        self.api_url =api_url # "http://109.228.59.105:9016/search" # Ensure this is the correct URL for your search service
        self.timeout = 30
        self.collection_name = collection_name
        self.document_contexts = {} # In-memory storage for session-specific document contexts
        prompts = load_prompts()
        self.context_request_analyzer_prompt = prompts.get('context_request_analyzer_prompt', '')
        self.context_relevance_prompt = prompts.get('context_relevance_prompt', '')
        self.context_advisor_prompt = prompts.get('context_advisor_prompt', '')
        self.async_client = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True 
        )
    def _get_embedding(self, text: str) -> np.ndarray:
        # This method does not perform I/O, so it remains synchronous.
        if not text: return np.zeros(10)
        vec = np.array([ord(c) for c in text[:min(len(text), 100)]])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    async def analyze_document_for_session(self, work_notes_manager: WorkNotesManager, document_id: str, text: str):
        """
        Analyzes a document and stores its context in memory for the current session.
        This remains a standard async function, adding notes to WorkNotesManager, not yielding.
        """
        logger.info(f"Analyzing document for in-session context: {document_id}")
        summary = f"Content from document '{document_id}': {text[:1000]}..."
        self.document_contexts[document_id] = {"document_id": document_id, "context_summary": summary}
        work_notes_manager.add_note("ContextualAgent", f"Document `{document_id}` analyzed and context stored for this session.")
        # No yield here as per current design choice for non-streaming internal steps.

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    # MODIFIED: Changed return type to AsyncIterator to yield notes and then a final result
    async def is_context_required(self, work_notes_manager: WorkNotesManager, query: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Determines if the given query requires additional context, yielding notes and the boolean result.
        """
        note_content = f"Analyzing if query '{query}' requires further context."
        work_notes_manager.add_note("ContextualAgent", note_content)
        yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
        
        prompt = ChatPromptTemplate.from_template(self.context_request_analyzer_prompt)
        chain = prompt | self.llm | StrOutputParser()
        
        is_required = False # Initialize result
        try:
            # FIX: Use await chain.ainvoke() for asynchronous LLM interaction
            response = await chain.ainvoke({"query": query})
            note_content = f"Context Agent (is_context_required): '{response}'"
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
            is_required = "yes" in response.lower()
        except Exception as e:
            logger.error(f"Context request analysis failed for query '{query}': {e}", exc_info=True)
            note_content = f"ERROR: Context analysis failed. Error: {str(e)}"
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
            is_required = False

        # Yield the final result for this operation
        yield {"type": "context_required_result", "content": is_required}

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    # MODIFIED: Changed return type to AsyncIterator to yield notes and then a final result
    async def is_context_relevant(self, work_notes_manager: WorkNotesManager, query: str, advice: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Validates if the found context is relevant to the original query, yielding notes and the boolean result.
        """
        if not advice or "No relevant context" in advice: 
            note_content = "No advice provided or deemed irrelevant by content."
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
            yield {"type": "context_relevant_result", "content": False}
            return
            
        note_content = f"Validating if the context is relevant to the query."
        work_notes_manager.add_note("ContextualAgent", note_content)
        yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}

        prompt = ChatPromptTemplate.from_template(self.context_relevance_prompt)
        chain = prompt | self.llm | StrOutputParser()
        
        is_relevant_bool = False # Initialize result
        try:
            # FIX: Use await chain.ainvoke() for asynchronous LLM interaction
            response = await chain.ainvoke({"query": query, "advice": advice})
            note_content = f"Context Agent (is_context_relevant): '{response}'"
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
            is_relevant_bool = "yes" in response.lower()
        except Exception as e:
            logger.error(f"Context relevance check failed for query '{query}': {e}", exc_info=True)
            note_content = f"ERROR: Context relevance check failed. Error: {str(e)}"
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
            is_relevant_bool = False

        # Yield the final result for this operation
        yield {"type": "context_relevant_result", "content": is_relevant_bool}

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    # MODIFIED: Changed return type to AsyncIterator to yield notes and then a final result
    async def find_and_interpret_context(self, work_notes_manager: WorkNotesManager, query: str, top_k: int = 3) -> AsyncIterator[Dict[str, Any]]:
        """
        Finds relevant context from the external search service and interprets it, yielding notes and the interpreted context.
        """
        note_content = f"Searching for and interpreting context for query: '{query}'."
        work_notes_manager.add_note("ContextualAgent", note_content)
        yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}

        api_results: List[Dict[str, Any]] = []
        try:
            payload = {"collection_name": self.collection_name, "query": query, "top_k": top_k}
            note_content = f"Calling external search service at {self.api_url} with payload: {payload}"
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}

            response = await self.async_client.post(self.api_url, json=payload)
            response.raise_for_status()
            api_results = response.json().get("results", [])
            note_content = f"Received {len(api_results)} results from external search service."
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
        except httpx.RequestError as e:
            logger.error(f"External search service request failed: {e}", exc_info=True)
            note_content = f"ERROR: External search service request failed: {str(e)}"
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
            # Yield empty/error context and return
            yield {"type": "interpreted_context_result", "content": "No relevant context was found due to search error."}
            return
        except Exception as e:
            logger.error(f"An unexpected error occurred during external search: {e}", exc_info=True)
            note_content = f"ERROR: An unexpected error occurred during external search: {str(e)}"
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
            # Yield empty/error context and return
            yield {"type": "interpreted_context_result", "content": "An error occurred while retrieving context."}
            return
        
        if not api_results: 
            note_content = "No relevant context was found from external search."
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
            yield {"type": "interpreted_context_result", "content": "No relevant context was found."}
            return
        
        clean_summary = f"found relevant document(s): {json.dumps(api_results)}"
        prompt = ChatPromptTemplate.from_template(self.context_advisor_prompt)
        chain = prompt | self.llm | StrOutputParser()
        
        interpreted_context_str = "" # Initialize result
        try:
            # FIX: Use await chain.ainvoke() for asynchronous LLM interaction
            interpreted_context_str = await chain.ainvoke({"query": query, "clean_context_summary": clean_summary})
            note_content = "Context interpreted successfully by LLM."
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
        except Exception as e:
            logger.error(f"Context interpretation by LLM failed: {e}", exc_info=True)
            note_content = f"ERROR: Context interpretation by LLM failed: {str(e)}"
            work_notes_manager.add_note("ContextualAgent", note_content)
            yield {"type": "work_note", "content": f"[ContextualAgent] {note_content}"}
            interpreted_context_str = "An error occurred while interpreting the context."
        
        # Yield the final interpreted context
        yield {"type": "interpreted_context_result", "content": interpreted_context_str}