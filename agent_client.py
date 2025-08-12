#agent_client.py
import os
import uvicorn
import json
import httpx
import logging
import yaml
from contextlib import asynccontextmanager
from typing import Dict, AsyncIterator, List

# FastAPI and Pydantic imports
from fastapi import FastAPI, Request, HTTPException, status, Depends, Security
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Imports from your project
from bowmen_agent import GRCPipeline
from langchain_ibm import ChatWatsonx
# --- Import all your agent classes ---
from Agents.router import RouterAgent
from Agents.tools_agent import Tools_Agent
from Agents.evidencer import Evidence_Agent
from Agents.answering_agent import AnswerAgent
from Agents.query_rephraser import QueryRephraser
from Agents.contexual_agent import ContextualAgent

# ==================== 1. Configuration ====================
def load_config(path: str = "config/config.yaml") -> dict:
    """Loads configuration from a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# ==================== 2. Pydantic Models ====================
class StartSessionRequest(BaseModel):
    session_id: str
    user_db_id: str

class IndexRequest(BaseModel):
    session_id: str
    filename: str
    base64_content: str
    user_db_id: str

class AskRequest(BaseModel):
    session_id: str
    question: str
    user_db_id: str
    has_document: bool

# ==================== 3. Security ====================
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Validates the API key from the request header against the config."""
    if not api_key_header or api_key_header != config['server']['api_key']:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key")
    return api_key_header

# ==================== 4. Logger ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agent_client")

# ==================== 5. FastAPI Application Setup ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy clients and stateless agents once on startup."""
    logger.info("🚀 Server starting up...")
    os.environ["WATSONX_APIKEY"] = config['llm']['api_key']
    
    # 1. Load the shared LLM Client
    llm_client = ChatWatsonx(
        model_id=config['llm']['model_id'],
        url=config['llm']['url'],
        project_id=config['llm']['project_id'],
        params={"temperature": 0}
    )
    app.state.llm_client = llm_client
    logger.info("✅ LLM client loaded.")

    # 2. Load all stateless agents that only depend on the LLM
    evidence_agent = Evidence_Agent(llm_client)
    app.state.router_agent = RouterAgent(llm_client)
    app.state.tools_agent = Tools_Agent(llm_client, evidence_agent) # Depends on LLM and evidence_agent
    app.state.rephraser_agent = QueryRephraser(llm_client)
    app.state.answer_agent = AnswerAgent(llm_client)
    logger.info("✅ All stateless agents initialized.")
    
    yield
    logger.info("🛑 Server shutting down.")

app = FastAPI(title="Scalable AI Agent Server", lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['server']['allowed_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 6. API Endpoints ====================

@app.post("/start_session", response_model=str, tags=["Session"])
async def start_session(req: StartSessionRequest, api_key: str = Depends(get_api_key)):
    logger.info(f"🔐 Session started: {req.session_id} for user {req.user_db_id}")
    return req.session_id

@app.post("/index_file", tags=["Document"])
async def index_file(req: IndexRequest, api_key: str = Depends(get_api_key)):
    collection_name = f"rfp_{req.session_id}"
    payload = {
        "collection_name": collection_name,
        "files": [{"filename": req.filename, "base64_content": req.base64_content}]
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                config['contexual_agent']['indexer_service_url'],
                json=payload,
                timeout=180
            )
            response.raise_for_status()
        logger.info(f"📄 File '{req.filename}' indexed for session {req.session_id}")
        return {"message": "File indexed successfully.", "session_id": req.session_id, "filename": req.filename}
    except Exception as e:
        logger.error(f"❌ Indexing failed for session {req.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="File indexing failed.")

@app.post("/ask_stream", tags=["AI"])
async def ask_stream(req: AskRequest, request: Request, api_key: str = Depends(get_api_key)):
    """
    Handles an AI question by assembling pre-loaded agents and streaming the response.
    """
    # ADDED: Log the incoming request to diagnose the flag issue
    logger.info(f"Received /ask_stream request for session {req.session_id}. Document flag is: {req.has_document}")
    
    # Create the session-specific ContextualAgent
    collection_name = f"rfp_{req.session_id}"
    context_agent = ContextualAgent(request.app.state.llm_client, collection_name, config['contexual_agent']['retriver_service_url'])

    # Assemble the pipeline with pre-loaded, shared agents from app.state
    pipeline = GRCPipeline(
        router_agent=request.app.state.router_agent,
        tools_agent=request.app.state.tools_agent,
        rephraser_agent=request.app.state.rephraser_agent,
        answer_agent=request.app.state.answer_agent,
        context_agent=context_agent  # Pass in the session-specific one
    )

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in pipeline.run(
                user_question=req.question,
                session_has_document=req.has_document
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"❌ Streaming error for session {req.session_id}: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': 'AI pipeline error.'})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ==================== 7. Run Server ====================
if __name__ == "__main__":
    logger.info("🔧 Starting AI Agent Server with config...")
    uvicorn.run("agent_client:app", host="0.0.0.0", port=9008, log_level="info", reload=True)