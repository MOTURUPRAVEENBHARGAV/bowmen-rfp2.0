import os
import uuid
import msal
import uvicorn
import asyncpg
import httpx
import json
import secrets

from fastapi import FastAPI, Request, HTTPException, Depends, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from typing import Optional
import asyncio
from datetime import datetime, timezone, timedelta
from fastapi.middleware.cors import CORSMiddleware


from config import DATABASE_URL  # Ensure you have a config.py with DATABASE_URL


# --- User Authentication & Session Config ---
CLIENT_ID = "df02b04d-753b-4b0e-a21b-a861f7e26423"
CLIENT_SECRET = "B558Q~dUBt1vHj1DgFRFo8EXqkp_tRmBw9G3fcoV"
TENANT_ID = "ac2b8ca8-067c-4035-8060-5e9e3ef79a45"
REDIRECT_URI = "http://localhost:9008/auth/microsoft/callback"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["User.Read"]
ALLOWED_MS_DOMAIN = "bowmengroup.com"

# --- Bommen Agent API Config ---
AGENT_API_URL = "https://bowmen-agent.1ws84ltl5maf.us-south.codeengine.appdomain.cloud"
AGENT_API_KEY = "Bwmn_sk_u8x5fE7Yt2RzP9wBvNcK6gH4jL1oS3aV"
app = FastAPI()

# --- ADD THIS CORS MIDDLEWARE BLOCK ---
# This must be placed right after `app = FastAPI()`
origins = [
    "http://localhost:5173", # The origin of your React app
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
# ------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
db_pool = None

# --- Database Table Creation ---
async def create_tables():
    """Creates all necessary database tables if they don't exist."""
    async with db_pool.acquire() as conn:
        # Users table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Web Sessions table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS web_sessions (
                session_id UUID PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
                user_data_json TEXT NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL
            );
        ''')
        
        # Chat Sessions table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id VARCHAR(40) PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                has_documents BOOLEAN DEFAULT FALSE
            );
        ''')
        
        # Chat Messages table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(40) REFERENCES chat_sessions(session_id) ON DELETE CASCADE NOT NULL,
                message_content TEXT NOT NULL,
                message_type VARCHAR(10) NOT NULL, -- 'user' or 'bot'
                work_notes TEXT DEFAULT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Session Files table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS session_files (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(40) REFERENCES chat_sessions(session_id) ON DELETE CASCADE NOT NULL,
                filename VARCHAR(255) NOT NULL,
                upload_status VARCHAR(20) NOT NULL, -- 'uploading', 'uploaded', 'failed'
                uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (session_id, filename)
            );
        ''')
        
        print("Database tables checked/created successfully.")

# --- Password & User Helpers ---
async def get_user_by_email(email: str):
    async with db_pool.acquire() as conn:
        return await conn.fetchrow("SELECT id, email, password_hash FROM users WHERE email = $1", email)

async def create_user(email: str, password_hash: str):
    async with db_pool.acquire() as conn:
        try:
            return await conn.fetchrow(
                "INSERT INTO users (email, password_hash) VALUES ($1, $2) RETURNING id, email",
                email, password_hash
            )
        except asyncpg.exceptions.UniqueViolationError:
            return None

async def verify_password(plain_password: str, hashed_password: str) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, pwd_context.verify, plain_password, hashed_password)

async def get_password_hash(password: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, pwd_context.hash, password)

# --- MSAL Helper ---
def _build_msal_app(cache=None):
    return msal.ConfidentialClientApplication(
        client_id=CLIENT_ID, client_credential=CLIENT_SECRET, authority=AUTHORITY, token_cache=cache
    )

# --- App Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)
        print("PostgreSQL connection pool created.")
        await create_tables()
    except Exception as e:
        print(f"FATAL: Failed to connect to PostgreSQL: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if db_pool:
        await db_pool.close()
        print("PostgreSQL connection pool closed.")

# --- DB-Backed Session Management ---
async def create_web_session(response: Response, user_record: dict, source: str, name: str = None, oid: str = None):
    session_id = uuid.uuid4()
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    user_data = {
        "id": user_record["id"], "email": user_record["email"], "name": name or user_record["email"].split("@")[0],
        "source": source, "oid": oid,
    }
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO web_sessions (session_id, user_id, user_data_json, expires_at) VALUES ($1, $2, $3, $4)",
            session_id, user_record["id"], json.dumps(user_data), expires_at
        )
    response.set_cookie(key="session_id", value=str(session_id), expires=expires_at, httponly=True, samesite="lax")

async def get_current_user(request: Request):
    session_id_str = request.cookies.get("session_id")
    if not session_id_str: raise HTTPException(status_code=401, detail="Not authenticated")
    try: session_id = uuid.UUID(session_id_str)
    except ValueError: raise HTTPException(status_code=401, detail="Invalid session identifier")
    async with db_pool.acquire() as conn:
        if (datetime.now().minute % 15) == 0: await conn.execute("DELETE FROM web_sessions WHERE expires_at < NOW()")
        session_record = await conn.fetchrow("SELECT user_data_json FROM web_sessions WHERE session_id = $1 AND expires_at >= NOW()", session_id)
    if not session_record: raise HTTPException(status_code=401, detail="Session expired or not found")
    return json.loads(session_record["user_data_json"])

# --- Authentication Routes (No Changes Needed Here) ---
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def handle_login(response: Response, email: str = Form(...), password: str = Form(...)):
    user_record = await get_user_by_email(email)
    if not user_record or not await verify_password(password, user_record["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    await create_web_session(response, user_record, "database")
    return {"status": "success", "redirect_url": "/chat"}

@app.post("/register")
async def handle_register(response: Response, email: str = Form(...), password: str = Form(...)):
    if not email or not password: raise HTTPException(status_code=400, detail="Email and password are required.")
    hashed_password = await get_password_hash(password)
    new_user = await create_user(email, hashed_password)
    if not new_user: raise HTTPException(status_code=409, detail="User with this email already exists.")
    await create_web_session(response, new_user, "database")
    return {"status": "success", "redirect_url": "/chat"}

@app.get("/auth/microsoft/login")
async def microsoft_login():
    msal_app = _build_msal_app()
    state = str(uuid.uuid4())
    response = RedirectResponse("/")
    response.set_cookie(key="oauth_state", value=state, max_age=600, httponly=True)
    auth_url = msal_app.get_authorization_request_url(scopes=SCOPE, state=state, redirect_uri=REDIRECT_URI)
    response.headers["Location"] = auth_url
    response.status_code = 307
    return response

@app.get("/auth/microsoft/callback")
async def microsoft_callback(request: Request, code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    if error: return RedirectResponse(f"/?error=Microsoft login failed: {error}")
    expected_state = request.cookies.get("oauth_state")
    if not expected_state or expected_state != state: return RedirectResponse(f"/?error=Invalid OAuth state")
    msal_app = _build_msal_app()
    result = msal_app.acquire_token_by_authorization_code(code, scopes=SCOPE, redirect_uri=REDIRECT_URI)
    if "error" in result: return RedirectResponse(f"/?error=Token acquisition failed: {result.get('error_description')}")
    claims = result.get("id_token_claims", {})
    user_email = claims.get("preferred_username")
    if not user_email or not user_email.lower().endswith(f"@{ALLOWED_MS_DOMAIN}"):
        return RedirectResponse(f"/?error=Access denied. Only @{ALLOWED_MS_DOMAIN} accounts are allowed.")
    user_record = await get_user_by_email(user_email)
    if not user_record:
        dummy_password_hash = await get_password_hash(str(uuid.uuid4()))
        user_record = await create_user(user_email, dummy_password_hash)
        if not user_record: return RedirectResponse(f"/?error=Failed to provision user account.")
    redirect_response = RedirectResponse("/chat")
    await create_web_session(redirect_response, user_record, "microsoft", claims.get("name"), claims.get("oid"))
    redirect_response.delete_cookie("oauth_state")
    return redirect_response

# --- Main Application Routes ---
@app.get("/chat", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: dict = Depends(get_current_user)):
    return templates.TemplateResponse("index.html", { "request": request, "user_name": current_user.get("name"), "user_email": current_user["email"] })

@app.get("/logout")
async def logout(request: Request):
    session_id_str = request.cookies.get("session_id")
    if session_id_str:
        try:
            session_id = uuid.UUID(session_id_str)
            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM web_sessions WHERE session_id = $1", session_id)
        except (ValueError, asyncpg.PostgresError) as e:
            print(f"Error during logout: {e}")
    response = RedirectResponse("/")
    response.delete_cookie("session_id")
    return response

# --- API Routes ---

@app.get("/api/check_auth")
async def check_auth_status(current_user: dict = Depends(get_current_user)):
    """
    An endpoint for the frontend to verify if a user's session cookie is valid.
    The `get_current_user` dependency handles all the validation.
    If the dependency passes, it returns a 200 OK. If not, it raises a 401.
    """
    return {"status": "authenticated", "user": current_user}

# NEW HELPER FUNCTION
async def _get_or_create_session(conn, session_id: Optional[str], user_id: int):
    """
    Checks if a session exists and belongs to the user. If session_id is None,
    it creates a new session in the database and returns the new ID.
    Returns a tuple: (the_session_id, was_created_boolean)
    """
    if session_id:
        # If a session ID is provided, verify it exists and belongs to the user.
        record = await conn.fetchval("SELECT session_id FROM chat_sessions WHERE session_id = $1 AND user_id = $2", session_id, user_id)
        if record:
            return record, False # Return existing ID, was_created is False
    
    # If no session_id or it's invalid, create a new one.
    new_session_id = f"chat_{secrets.token_hex(16)}"
    
    # Call the agent API to initialize the session there as well
    payload_to_agent = {"session_id": new_session_id, "user_db_id": str(user_id)}
    try:
        async with httpx.AsyncClient() as client:
            agent_response = await client.post(f"{AGENT_API_URL}/start_session", headers={"X-API-Key": AGENT_API_KEY}, json=payload_to_agent)
            agent_response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        detail = f"Agent service unavailable: {e}" if isinstance(e, httpx.RequestError) else f"Agent error: {e.response.text}"
        raise HTTPException(status_code=503, detail=detail)
        
    # Insert the new session into our database
    await conn.execute("INSERT INTO chat_sessions (session_id, user_id) VALUES ($1, $2)", new_session_id, user_id)
    print(f"Created new session {new_session_id} for user {user_id}")
    return new_session_id, True # Return new ID, was_created is True


@app.post("/api/start_session")
async def proxy_start_session(request: Request, current_user: dict = Depends(get_current_user)):
    """
    This endpoint is now ONLY for loading an existing session's history.
    It no longer creates new sessions.
    """
    frontend_payload = await request.json()
    session_id = frontend_payload.get("session_id")
    user_db_id = current_user["id"]
    
    if not session_id:
        # This case happens if the frontend asks to load a 'null' session, which means
        # it's a new chat. We just return an empty state without touching the DB.
        return {"session_id": None, "messages": [], "uploaded_files": [], "has_documents": False}

    async with db_pool.acquire() as conn:
        session_record = await conn.fetchrow("SELECT session_id, has_documents FROM chat_sessions WHERE session_id = $1 AND user_id = $2", session_id, user_db_id)
        
        if not session_record:
            raise HTTPException(status_code=404, detail="Chat session not found or you don't have permission.")

        messages = await conn.fetch("SELECT message_content, message_type, work_notes FROM chat_messages WHERE session_id = $1 ORDER BY timestamp ASC", session_id)
        files = await conn.fetch("SELECT filename, upload_status FROM session_files WHERE session_id = $1 ORDER BY uploaded_at ASC", session_id)
        await conn.execute("UPDATE chat_sessions SET last_active = NOW() WHERE session_id = $1", session_id)
        
        return {
            "session_id": session_id, 
            "messages": [dict(m) for m in messages], 
            "uploaded_files": [dict(f) for f in files], 
            "has_documents": session_record["has_documents"]
        }


@app.get("/api/chat_sessions")
async def get_chat_sessions(current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        sessions = await conn.fetch("""
            SELECT cs.session_id, cs.last_active, cs.has_documents,
                   (SELECT cm.message_content FROM chat_messages cm WHERE cm.session_id = cs.session_id AND cm.message_type = 'user' ORDER BY cm.timestamp ASC LIMIT 1) AS first_user_message
            FROM chat_sessions cs WHERE cs.user_id = $1 ORDER BY cs.last_active DESC
            """, current_user["id"])
        return [{"session_id": s["session_id"], "first_message": s["first_user_message"] or "New Chat", "last_active": s["last_active"].isoformat(), "has_documents": s["has_documents"]} for s in sessions]


@app.post("/api/index_file")
async def proxy_index_file(request: Request, current_user: dict = Depends(get_current_user)):
    frontend_payload = await request.json()
    session_id = frontend_payload.get("session_id") # Can be null
    filename = frontend_payload.get("filename")
    
    if not all([filename, frontend_payload.get("base64_content")]):
        raise HTTPException(status_code=400, detail="Missing filename or file content.")
    
    async with db_pool.acquire() as conn:
        # If session_id is null, this will create a new one. Otherwise, it verifies ownership.
        session_id, was_created = await _get_or_create_session(conn, session_id, current_user["id"])
        
        await conn.execute("INSERT INTO session_files (session_id, filename, upload_status) VALUES ($1, $2, 'uploading') ON CONFLICT (session_id, filename) DO UPDATE SET upload_status = 'uploading', uploaded_at = NOW()", session_id, filename)

    agent_payload = {"session_id": session_id, "user_db_id": str(current_user["id"]), "filename": filename, "base64_content": frontend_payload["base64_content"]}
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{AGENT_API_URL}/index_file", headers={"X-API-Key": AGENT_API_KEY}, json=agent_payload)
            response.raise_for_status()
        
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE session_files SET upload_status = 'uploaded' WHERE session_id = $1 AND filename = $2", session_id, filename)
            await conn.execute("UPDATE chat_sessions SET has_documents = TRUE, last_active = NOW() WHERE session_id = $1", session_id)
        
        # Return the session_id so the frontend knows it if it was just created
        return {"status": "success", "message": f"File {filename} indexed.", "session_id": session_id}
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE session_files SET upload_status = 'failed' WHERE session_id = $1 AND filename = $2", session_id, filename)
        detail = f"Agent service unavailable: {e}" if isinstance(e, httpx.RequestError) else f"Agent error: {e.response.text}"
        status_code = 503 if isinstance(e, httpx.RequestError) else e.response.status_code
        raise HTTPException(status_code=status_code, detail=detail)

@app.post("/api/ask_stream")
async def proxy_ask_stream(request: Request, current_user: dict = Depends(get_current_user)):
    frontend_payload = await request.json()
    session_id = frontend_payload.get("session_id") # Can be null for a new chat
    question = frontend_payload.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Missing question.")

    async def stream_generator():
        nonlocal session_id # Allow modification of the outer scope session_id
        bot_response_text = ""
        bot_work_notes = []
        
        try:
            # Step 1: Get or create the session using your helper function.
            async with db_pool.acquire() as conn:
                session_id, was_created = await _get_or_create_session(conn, session_id, current_user["id"])

                # If the session was just created, send its ID to the client immediately.
                if was_created:
                    created_event = {"type": "session_created", "session_id": session_id}
                    yield f"data: {json.dumps(created_event)}\n\n".encode('utf-8')

                # Save the user's message.
                session_record = await conn.fetchrow("SELECT has_documents FROM chat_sessions WHERE session_id = $1", session_id)
                await conn.execute("INSERT INTO chat_messages (session_id, message_content, message_type) VALUES ($1, $2, 'user')", session_id, question)
                await conn.execute("UPDATE chat_sessions SET last_active = NOW() WHERE session_id = $1", session_id)

            # Step 2: Stream the agent's response while buffering it.
            agent_payload = {"session_id": session_id, "user_db_id": str(current_user["id"]), "question": question, "has_document": session_record["has_documents"]}
            headers = {"X-API-Key": AGENT_API_KEY, "Content-Type": "application/json"}
            endpoint = f"{AGENT_API_URL}/ask_stream"
            
            buffer = b''
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", endpoint, headers=headers, json=agent_payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
                        buffer += chunk

            # Step 3: After streaming, parse the complete buffer to reliably find the final answer.
            events = buffer.split(b'\n\n')
            for event in events:
                if event.startswith(b'data:'):
                    try:
                        event_str = event.replace(b'data:', b'', 1).strip()
                        if not event_str: continue
                        event_data = json.loads(event_str)
                        if event_data.get('type') == 'done':
                            bot_response_text = event_data.get('answer', '')
                            bot_work_notes = event_data.get('work_notes', [])
                            break
                    except (json.JSONDecodeError, IndexError):
                        continue
        
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            detail = f"Agent service unavailable: {e}" if isinstance(e, httpx.RequestError) else f"Agent error: {e.response.text}"
            bot_response_text = f"Error: {detail}"
            error_event = {"type": "error", "content": detail}
            yield f"data: {json.dumps(error_event)}\n\n".encode('utf-8')
        
        finally:
            if bot_response_text:
                async with db_pool.acquire() as conn:
                    work_notes_json = json.dumps(bot_work_notes) if bot_work_notes else None
                    await conn.execute(
                        "INSERT INTO chat_messages (session_id, message_content, message_type, work_notes) VALUES ($1, $2, 'bot', $3)",
                        session_id, bot_response_text, work_notes_json
                    )
                    await conn.execute("UPDATE chat_sessions SET last_active = NOW() WHERE session_id = $1", session_id)
                    print(f"[DB_SAVE] Successfully saved response for session {session_id} to database.")
            else:
                print(f"[DB_SAVE_WARNING] No final answer was captured for session {session_id}. Nothing to save.")

    return StreamingResponse(stream_generator(), headers={
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no" # This header is crucial for proxies like Nginx
    })


# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists("templates"): os.makedirs("templates")
    if not os.path.exists("static"): os.makedirs("static")
    uvicorn.run("main:app", host="0.0.0.0", port=9008, reload=True)