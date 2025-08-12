
import os
import re
import json
import yaml
import asyncio
from typing import Dict, Any, Tuple, List, AsyncIterator

from langchain_ibm import ChatWatsonx
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import BooleanOutputParser 

from config.logger_config import get_logger, WorkNotesManager
from Agents.router import RouterAgent
from Agents.tools_agent import Tools_Agent
from Agents.evidencer import Evidence_Agent
from Agents.answering_agent import AnswerAgent
from Agents.query_rephraser import QueryRephraser
from Agents.contexual_agent import ContextualAgent

logger = get_logger("bowmen_agent_pipeline")

def load_yaml_config(path: str = "config/config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

config = load_yaml_config()

class GRCPipeline:
    def __init__(
        self,
        router_agent: RouterAgent,
        tools_agent: Tools_Agent,
        rephraser_agent: QueryRephraser,
        answer_agent: AnswerAgent,
        context_agent: ContextualAgent
    ):
        logger.info(f"Initializing GRCPipeline for collection: {context_agent.collection_name}")
        self.MAX_REFINEMENTS = 3
        self.router_agent = router_agent
        self.tools_agent = tools_agent
        self.rephraser_agent = rephraser_agent
        self.answer_agent = answer_agent
        self.context_agent = context_agent
        self.evidence_agent = self.tools_agent.evidence_agent
        self.MAX_QUERY_LENGTH = 500
        self.MAX_QUERY_PARTS = 2
        self.POINT_PATTERN = re.compile(r'(\b[i1a]\.|\b[ii2b]\.|\b[iii3c]\.)', re.IGNORECASE)

    def _create_note_event(self, work_notes_manager: WorkNotesManager, source: str, message: str) -> Dict[str, str]:
        work_notes_manager.add_note(source, message)
        return {"type": "work_note", "content": f"[{source}] {message}"}

    def _is_complex_query(self, query: str) -> bool:
            """
            Determines if a query is complex based on its length, structure, or content.

            A query is considered complex if it meets any of the following criteria:
            1. The total number of characters exceeds self.MAX_QUERY_LENGTH.
            2. It contains more than 20 words.
            3. It includes newline characters ('\n'), suggesting multiple parts.
            4. It contains patterns that indicate enumerated points (e.g., 'i.', 'a.', '1.').
            """
            # Check for more than 20 words
            if len(query.split()) > 20:
                return True
            
            # Check for newline characters
            if '\n' in query:
                return True

            # Original check for overall character length
            if len(query) > self.MAX_QUERY_LENGTH:
                return True

            # Original check for enumerated points
            return bool(self.POINT_PATTERN.search(query))

    def _is_version_query(self, query: str) -> bool:
        return any(keyword in query.lower() for keyword in ["version", "latest version", "release"])

    async def _search_and_see(self, work_notes_manager: WorkNotesManager, question: str, seen_questions: set, active_context: str) -> AsyncIterator[Dict[str, Any]]:
        if question in seen_questions:
            yield self._create_note_event(work_notes_manager, "GRCPipeline", f"Loop Prevention: Already processed '{question}'.")
            yield {"type": "search_results", "content": {"vectordb_results": [], "websearch_results": [], "urls": []}}
            return

        seen_questions.add(question)
        yield self._create_note_event(work_notes_manager, "GRCPipeline", f"Initiating search for '{question}'.")

        vectordb_results: List[str] = []
        websearch_results: List[str] = []
        retrieved_urls: List[str] = []

        async for tool_event in self.tools_agent.search_tools(work_notes_manager, question, active_context):
            if tool_event["type"] == "work_note":
                yield tool_event
            elif tool_event["type"] == "tools_agent_results":
                vectordb_results.extend(tool_event["content"]["vectordb_results"])
                websearch_results.extend(tool_event["content"]["web_results"])
                retrieved_urls.extend(tool_event["content"]["urls"])

        yield self._create_note_event(work_notes_manager, "GRCPipeline", f"Search completed. Found {len(vectordb_results)} vectorDB and {len(websearch_results)} web search evidence items.")
        
        yield {"type": "search_results", "content": {
            "vectordb_results": vectordb_results,
            "websearch_results": websearch_results,
            "urls": retrieved_urls
        }}

    async def run(self, user_question: str, session_has_document: bool = False) -> AsyncIterator[Dict[str, Any]]:
        logger.info(f"GRCPipeline run started. Question: '{user_question[:50]}...'. Received session_has_document flag: {session_has_document}")
        
        work_notes_manager = WorkNotesManager()
        yield self._create_note_event(work_notes_manager, "GRCPipeline", f"Starting new run for question: '{user_question}'")

        evidences: List[str] = []
        seen_questions: set = set()
        active_context_str: str = ""
        all_urls: List[str] = []
        all_metadata: List[Dict[str, str]] = []

        if session_has_document:
            yield self._create_note_event(work_notes_manager, "GRCPipeline", "Checking for session context requirement.")
            
            context_required = False
            async for event in self.context_agent.is_context_required(work_notes_manager, user_question):
                if event["type"] == "work_note":
                    yield event
                elif event["type"] == "context_required_result":
                    context_required = event["content"]

            if context_required:
                yield self._create_note_event(work_notes_manager, "GRCPipeline", "Context required. Finding and interpreting.")
                
                advice_str = ""
                async for event in self.context_agent.find_and_interpret_context(work_notes_manager, user_question):
                    if event["type"] == "work_note":
                        yield event
                    elif event["type"] == "interpreted_context_result":
                        advice_str = event["content"]

                context_is_relevant = False
                async for event in self.context_agent.is_context_relevant(work_notes_manager, user_question, advice_str):
                    if event["type"] == "work_note":
                        yield event
                    elif event["type"] == "context_relevant_result":
                        context_is_relevant = event["content"]

                if context_is_relevant:
                    yield self._create_note_event(work_notes_manager, "GRCPipeline", "Context found and relevant.")
                    active_context_str = advice_str
                else:
                    yield self._create_note_event(work_notes_manager, "GRCPipeline", "Found context but deemed not relevant.")
            else:
                yield self._create_note_event(work_notes_manager, "GRCPipeline", "Context not required for this query.")

        yield self._create_note_event(work_notes_manager, "GRCPipeline", "Calling Router Agent...")
        
        router_response = {}
        async for router_event in self.router_agent.route_query(work_notes_manager, user_question, context=active_context_str):
            if router_event["type"] == "work_note":
                yield router_event
            elif router_event["type"] == "router_result":
                router_response = router_event["content"]

        if router_response.get("decision") != "ACCEPT":
            final_message = router_response.get("final_answer", "I am sorry, but I cannot answer that question based on my guidelines.")
            yield self._create_note_event(work_notes_manager, "GRCPipeline", f"Router provided a direct answer. Ending run.")
            yield {
                "type": "done",
                "answer": final_message,
                "work_notes": work_notes_manager.get_all_notes(),
                "urls": [],
                "metadata": [],
            }
            return

        current_questions = router_response.get("question", user_question)
        if isinstance(current_questions, str):
            current_questions = [current_questions]
        elif self._is_complex_query(user_question):
            yield self._create_note_event(work_notes_manager, "GRCPipeline", "Query is complex or contains enumerated points. Decomposing with QueryRephraser.")
            async for rephraser_event in self.rephraser_agent.decompose_query(work_notes_manager, user_question):
                if rephraser_event["type"] == "work_note":
                    yield rephraser_event
                elif rephraser_event["type"] == "decomposer_result":
                    current_questions = rephraser_event["content"].get("rephrased_question", [user_question])

        vectorbd_res_str_initial: List[str] = []
        websearch_res_initial: List[str] = []
        retrieved_urls_initial: List[str] = []

        for question in current_questions[:3]:
            async for search_event in self._search_and_see(work_notes_manager, question, seen_questions, active_context_str):
                if search_event["type"] == "work_note":
                    yield search_event
                elif search_event["type"] == "search_results":
                    vectorbd_res_str_initial.extend(search_event["content"]["vectordb_results"])
                    websearch_res_initial.extend(search_event["content"]["websearch_results"])
                    retrieved_urls_initial.extend(search_event["content"]["urls"])

        evidences.extend(vectorbd_res_str_initial)
        evidences.extend(websearch_res_initial)
        all_urls.extend(retrieved_urls_initial)
        all_metadata.extend([{"text": item} for item in vectorbd_res_str_initial])
        all_metadata.extend([{"text": item} for item in websearch_res_initial])

        refinement_count = 0
        while refinement_count < self.MAX_REFINEMENTS:
            if not evidences:
                feedback_thought = "The initial search failed to find any information. Create a more effective search query."
                yield self._create_note_event(work_notes_manager, "GRCPipeline", "Thought: Initial search found no evidence. Activating Query Rephraser.")
            else:
                yield self._create_note_event(work_notes_manager, "GRCPipeline", "Action: Performing holistic gap analysis on all collected evidence.")
                combined_evidence_for_analysis = "\n\n---\n\n".join(list(set(evidences)))
                
                analysis_result = {}
                async for analysis_event in self.evidence_agent.run_gap_analysis(work_notes_manager, user_question, combined_evidence_for_analysis, active_context_str):
                    if analysis_event["type"] == "work_note":
                        yield analysis_event
                    elif analysis_event["type"] == "gap_analysis_result":
                        analysis_result = analysis_event["content"]
                
                sufficiency = analysis_result.get("Sufficient", "false")
                feedback_thought = analysis_result.get("Thought", "")

                if sufficiency == "true":
                    yield self._create_note_event(work_notes_manager, "GRCPipeline", "Thought: Evidence is sufficient. Generating a final answer")
                    break

                if self._is_version_query(user_question) and "version" not in combined_evidence_for_analysis.lower():
                    feedback_thought = "The evidence does not explicitly provide the required details, requiring a more targeted search."
                    sufficiency = "undetermined"

                refinement_count += 1
                yield self._create_note_event(work_notes_manager, "GRCPipeline", f"Thought: Evidence is insufficient. Invoking Query Rephraser (Attempt {refinement_count}/{self.MAX_REFINEMENTS}).")
                
                new_questions = [""]
                if self._is_complex_query(user_question):
                    async for rephraser_event in self.rephraser_agent.decompose_query(work_notes_manager, user_question):
                        if rephraser_event["type"] == "work_note":
                            yield rephraser_event
                        elif rephraser_event["type"] == "decomposer_result":
                            new_questions = rephraser_event["content"].get("rephrased_question", [""])
                else:
                    async for rephraser_event in self.rephraser_agent.refine_query(work_notes_manager, user_question, feedback_thought, active_context_str):
                        if rephraser_event["type"] == "work_note":
                            yield rephraser_event
                        elif rephraser_event["type"] == "rephrased_question":
                            new_questions = [rephraser_event["content"]]

                for new_question in new_questions[:3]:
                    if new_question and new_question not in seen_questions:
                        new_vectorbd_res_str: List[str] = []
                        new_websearch_res: List[str] = []
                        new_retrieved_urls: List[str] = []

                        async for search_event_refined in self._search_and_see(work_notes_manager, new_question, seen_questions, active_context_str):
                            if search_event_refined["type"] == "work_note":
                                yield search_event_refined
                            elif search_event_refined["type"] == "search_results":
                                new_vectorbd_res_str = search_event_refined["content"]["vectordb_results"]
                                new_websearch_res = search_event_refined["content"]["websearch_results"]
                                new_retrieved_urls = search_event_refined["content"]["urls"]

                        new_evidence = new_vectorbd_res_str + new_websearch_res
                        if new_evidence:
                            evidences.extend(new_evidence)
                            all_metadata.extend([{"text": item} for item in new_vectorbd_res_str])
                            all_metadata.extend([{"text": item} for item in new_websearch_res])
                            all_urls.extend(new_retrieved_urls)
                        else:
                            yield self._create_note_event(work_notes_manager, "GRCPipeline", "Observation: Refined search did not provide new evidence.")
                    else:
                        yield self._create_note_event(work_notes_manager, "GRCPipeline", "Observation: Refiner did not produce a new, unique question. Halting refinement.")
                        break

        yield self._create_note_event(work_notes_manager, "GRCPipeline", "All search cycles complete. Proceeding to final answer generation.")

        final_answer_text: str = ""
        combined_evidence: str = ""

        if not evidences:
            final_answer_text = "I could not find much information to answer your question."
            combined_evidence = ""
        else:
            combined_evidence = "\n\n---\n\n".join(list(set(evidences)))
            
            answer_obj_data = {}
            async for answer_event in self.answer_agent.make_final_answer(work_notes_manager, user_question, combined_evidence, active_context_str):
                if answer_event["type"] == "work_note":
                    yield answer_event
                elif answer_event["type"] == "final_answer_content":
                    answer_obj_data = answer_event["content"]

            final_answer_text = answer_obj_data.get("final_answer", "Could not generate a final answer.")
            
        unique_urls = list(set(all_urls))
        
        seen_metadata_str = set()
        unique_metadata_list: List[Dict[str, str]] = []
        for d in all_metadata:
            d_to_sort = d.copy()
            if 'text' not in d_to_sort:
                d_to_sort['text'] = ''
            sorted_d_str = json.dumps(d_to_sort, sort_keys=True)
            
            if sorted_d_str not in seen_metadata_str:
                seen_metadata_str.add(sorted_d_str)
                unique_metadata_list.append(d)

        yield {
            "type": "done",
            "query": user_question,
            "answer": final_answer_text,
            "work_notes": work_notes_manager.get_all_notes(),
            "urls": unique_urls,
            "metadata": unique_metadata_list,
        }
