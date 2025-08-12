import os
import json
import httpx
import asyncio
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from config.logger_config import get_logger, WorkNotesManager
import yaml
from typing import List, Dict, Tuple, Any, AsyncIterator

logger = get_logger("tools_agent")

# --- Configuration Loading (EXACTLY AS PROVIDED BY USER IN PREVIOUS TOOLS_AGENT.PY) ---
class ConfigLoader:
    @staticmethod
    def load_config(config_path="config/config.yaml"):
        try:
            with open(config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_path}. Please create it.")
            # Provide a minimal fallback structure for config to prevent immediate crashes
            return {
                "google_api_keys": ["YOUR_GOOGLE_API_KEY_HERE"],
                "google_search_engine_id": "YOUR_CSE_ID_HERE", 
                "rfp_agent": {
                    "retriver_service_url": "http://localhost:8000/retrieve", # Example URL
                    "collection_name": "default_collection"
                }
            }
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {config_path}: {e}")
            return {
                "google_api_keys": ["YOUR_GOOGLE_API_KEY_HERE"],
                "google_search_engine_id": "YOUR_CSE_ID_HERE", # Reverted to exact original
                "rfp_agent": {
                    "retriver_service_url": "http://localhost:8000/retrieve",
                    "collection_name": "default_collection"
                }
            }
        except Exception as e:
            logger.error(f"An unexpected error occurred loading config: {e}")
            return {
                "google_api_keys": ["YOUR_GOOGLE_API_KEY_HERE"],
                "google_search_engine_id": "YOUR_CSE_ID_HERE", # Reverted to exact original
                "rfp_agent": {
                    "retriver_service_url": "http://localhost:8000/retrieve",
                    "collection_name": "default_collection"
                }
            }

config = ConfigLoader.load_config()
# --- END Configuration Loading ---


class Tools_Agent:
    HEADERS = {"User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0")}
    REQUEST_TIMEOUT = 30
    DISALLOWED_EXTENSIONS = {'.pdf', '.zip', '.jpg', '.png', '.gif', '.mp4', '.avi', '.mov'}

    def __init__(self, llm: Any, evidence_agent: Any):
        self.llm = llm
        self.evidence_agent = evidence_agent
        self.api_keys = config["google_api_keys"]
        self.cse_id = config["google_search_engine_id"]
        self.async_client = httpx.AsyncClient(headers=self.HEADERS, timeout=self.REQUEST_TIMEOUT)

    async def _post_vector_request(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        url = config['rfp_agent']['retriver_service_url']
        payload = {"collection_name": config['rfp_agent']['collection_name'], "question": question, "top_k": top_k}
        
        try:
            logger.info(f"Sending payload to VectorDB: {json.dumps(payload)}")
            response = await self.async_client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("results", [])
        except httpx.RequestError as e:
            logger.error(f"VectorDB request failed: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during VectorDB request: {e}", exc_info=True)
            return []

    async def vectordb(self, work_notes_manager: WorkNotesManager, question: str, context: str = None, top_k: int = 5, max_results: int = 5) -> AsyncIterator[Dict[str, Any]]:
        note_content = f"Searching internal Database for '{question}'..."
        work_notes_manager.add_note("ToolsAgent", note_content)
        yield {"type": "work_note", "content": f"[ToolsAgent] {note_content}"}

        chunks = await self._post_vector_request(question, top_k)
        if not chunks:
            note_content = "Observation: Internal VectorDB returned 0 results."
            work_notes_manager.add_note("ToolsAgent", note_content)
            yield {"type": "work_note", "content": f"[ToolsAgent] {note_content}"}
            yield {"type": "vectordb_results", "content": []}
            return

        relevant_chunks = [chunk.get("content", "") for chunk in chunks if chunk.get("content")]
        note_content = f"Observation: Found {len(relevant_chunks)} chunks in VectorDB."
        work_notes_manager.add_note("ToolsAgent", note_content)
        yield {"type": "work_note", "content": f"[ToolsAgent] {note_content}"}
        
        yield {"type": "vectordb_results", "content": relevant_chunks[:max_results]}

    async def do_webscraping(self, link: str) -> str | None:
        try:
            loader = AsyncHtmlLoader([link])
            docs = await loader.aload()
            if docs:
                return Html2TextTransformer().transform_documents(docs)[0].page_content
            return None
        except Exception as e:
            logger.exception(f"Web scraping failed for {link}: {e}")
            return None

    async def search_google(self, work_notes_manager: WorkNotesManager, query: str) -> List[str]:
        note_content = f"Searching Web for '{query}'..."
        work_notes_manager.add_note("ToolsAgent", note_content)
        
        for api_key in self.api_keys:
            try:
                url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={self.cse_id}"
                response = await self.async_client.get(url)
                response.raise_for_status()
                json_data = response.json()
                if "items" in json_data:
                    links = [item['link'] for item in json_data["items"]]
                    work_notes_manager.add_note("ToolsAgent", f"Google search found {len(links)} results.")
                    return links[:5]
            except httpx.RequestError as e:
                logger.warning(f"Google search error with one key ({api_key}): {e}")
            except Exception as e:
                logger.warning(f"An unexpected error occurred during Google search: {e}")
        
        work_notes_manager.add_note("ToolsAgent", "Observation: Google search returned 0 results or all API keys failed.")
        return []

    async def _scrape_and_filter_url(self, work_notes_manager: WorkNotesManager, question: str, url: str, context: str = None) -> Dict[str, str] | None:
        content = await self.do_webscraping(url)
        if not content:
            return None
        
        is_relevant = False
        # Consume the evidence_agent's generator for relevance check
        async for event in self.evidence_agent.run_quick_relevance_check(work_notes_manager, question, content, context=context):
            if event["type"] == "work_note":
                # Notes from Evidence_Agent's relevance check are added to WorkNotesManager
                # but not re-yielded directly by _scrape_and_filter_url as it's not a generator itself.
                pass 
            elif event["type"] == "relevance_check_result":
                is_relevant = event["content"]
        
        if is_relevant:
            work_notes_manager.add_note("ToolsAgent", f"Web search success for: {url}")
            return {"url": url, "content": content}
        return None

    async def scrape_ibm_docs(self, work_notes_manager: WorkNotesManager, question: str, context: str = None, top_k: int = 5, max_pages: int = 2) -> AsyncIterator[Dict[str, Any]]:
        urls = await self.search_google(work_notes_manager, question)
        
        note_content = f"Google search results: {len(urls)} URLs found."
        work_notes_manager.add_note("ToolsAgent", note_content)
        yield {"type": "work_note", "content": f"[ToolsAgent] {note_content}"}

        filtered_urls = [url for url in urls if not any(url.lower().endswith(ext) for ext in self.DISALLOWED_EXTENSIONS)]
        
        note_content = f"Starting parallel scraping of up to {max_pages} pages..."
        work_notes_manager.add_note("ToolsAgent", note_content)
        yield {"type": "work_note", "content": f"[ToolsAgent] {note_content}"}

        tasks = [self._scrape_and_filter_url(work_notes_manager, question, url, context) for url in filtered_urls[:max_pages]]
        relevant_results = await asyncio.gather(*tasks)
        
        relevant_results = [r for r in relevant_results if r is not None]

        web_contents = [r["content"] for r in relevant_results]
        web_urls = [r["url"] for r in relevant_results]

        note_content = f"Scraped and filtered {len(web_contents)} relevant web pages."
        work_notes_manager.add_note("ToolsAgent", note_content)
        yield {"type": "work_note", "content": f"[ToolsAgent] {note_content}"}
        
        yield {"type": "web_search_results", "content": {"web_contents": web_contents, "web_urls": web_urls}}


    # NEW HELPER: For consuming an async generator and pushing its items to a queue
    async def _queue_producer(self, generator: AsyncIterator[Dict[str, Any]], queue: asyncio.Queue, source_label: str):
        try:
            async for item in generator:
                if item["type"] == "work_note":
                    # Add source_label to content for clarity when streamed
                    content_prefix = f"[{source_label}] "
                    if not item["content"].startswith(content_prefix):
                        item["content"] = content_prefix + item["content"]
                await queue.put(item)
        except Exception as e:
            logger.error(f"Error in _queue_producer for {source_label}: {e}", exc_info=True)
            await queue.put({"type": "error", "source": source_label, "message": str(e)})
        finally:
            # Signal that this producer is done by putting a special "DONE" marker
            await queue.put({"type": "producer_done", "source": source_label})


    async def search_tools(self, work_notes_manager: WorkNotesManager, question: str, context: str = None, top_k: int = 5, max_results: int = 5) -> AsyncIterator[Dict[str, Any]]:
        """
        Orchestrates parallel asynchronous searches using VectorDB and web scraping,
        yielding notes from sub-tools and then the combined final result.
        This now uses an asyncio.Queue for truly interleaved streaming.
        """
        note_content = f"Initiating search using Tools for '{question}'."
        work_notes_manager.add_note("ToolsAgent", note_content)
        yield {"type": "work_note", "content": f"[ToolsAgent] {note_content}"}

        vectordb_results_list: List[str] = []
        web_contents_list: List[str] = []
        web_urls_list: List[str] = []

        event_queue = asyncio.Queue()
        num_producers = 2 # VectorDB and WebSearch

        # Start the producers (tasks that feed the queue)
        producers = [
            asyncio.create_task(self._queue_producer(
                self.vectordb(work_notes_manager, question, context, top_k, max_results),
                event_queue, "VectorDB"
            )),
            asyncio.create_task(self._queue_producer(
                self.scrape_ibm_docs(work_notes_manager, question, context, top_k, 5),
                event_queue, "WebSearch"
            )),
        ]

        completed_producers = 0
        while completed_producers < num_producers:
            item = await event_queue.get()
            event_queue.task_done() # Mark the task as done for the queue

            if item["type"] == "work_note":
                yield item # Yield work notes immediately as they arrive
            elif item["type"] == "producer_done":
                completed_producers += 1
            elif item["type"] == "error":
                yield {"type": "work_note", "content": f"[ERROR] {item['source']} producer error: {item['message']}"}
                logger.error(f"Producer error from {item['source']}: {item['message']}")
            else:
                # This must be a final result from one of the sub-generators
                if item["type"] == "vectordb_results":
                    vectordb_results_list.extend(item["content"])
                elif item["type"] == "web_search_results":
                    web_contents_list.extend(item["content"]["web_contents"])
                    web_urls_list.extend(item["content"]["web_urls"])
        
        # Ensure all producer tasks are awaited after consuming queue, for proper cleanup
        await asyncio.gather(*producers)


        note_content = f"Returning {len(vectordb_results_list)} VectorDB chunks and {len(web_contents_list)} web chunks."
        work_notes_manager.add_note("ToolsAgent", note_content)
        yield {"type": "work_note", "content": f"[ToolsAgent] {note_content}"}

        # Yield the final consolidated result for search_tools
        yield {"type": "tools_agent_results", "content": {
            "vectordb_results": vectordb_results_list,
            "web_results": web_contents_list,
            "urls": web_urls_list
        }}