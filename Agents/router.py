import json
import re
from config.logger_config import get_logger, WorkNotesManager
from langchain_ibm import ChatWatsonx
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.utils import load_prompts
from typing import AsyncIterator, Dict, Any

logger = get_logger("router_agent")

class RouterAgent:
    """
    A triage agent that detects code snippets and uses an LLM for all other routing decisions.
    """

    def __init__(self, llm: ChatWatsonx):
        self.llm = llm
        prompts = load_prompts()
        self.router_agent_prompt = prompts.get('router', '')
        # Regex to detect code snippets (backticks, programming keywords, JSON-like structures)
        

    async def route_query(self, work_notes_manager: WorkNotesManager, query: str, context: str = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Routes the user query, rejecting queries with code snippets or accepting for LLM processing.
        Yields work notes and the final result.
        """
        note_content = f"Routing query: '{query}'"
        work_notes_manager.add_note("RouterAgent", note_content)
        yield {"type": "work_note", "content": f"[RouterAgent] {note_content}"}



        # Step 2: Use LLM for nuanced routing (including greetings, rejections, and acceptance)
        prompt = ChatPromptTemplate.from_template(self.router_agent_prompt)
        chain = prompt | self.llm | JsonOutputParser()
        context_str = context if context else "No context provided."
        
        response = {}
        try:
            llm_response = await chain.ainvoke({"query": query, "context": context_str})
            response.update(llm_response)
        except Exception as e:
            logger.error(f"Router agent failed: {e}. Defaulting to REJECT.", exc_info=True)
            response = {
                "decision": "REJECT",
                "thought": "An internal error occurred during routing.",
                "question": query,
                "final_answer": "I'm sorry, I encountered an error and cannot process your request at this time."
            }
        
        note_content = f"Thought: {response.get('thought', 'No thought provided.')}"
        work_notes_manager.add_note("RouterAgent", note_content)
        yield {"type": "work_note", "content": f"[RouterAgent] {note_content}"}
        
        yield {"type": "router_result", "content": response}