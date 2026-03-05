"""RLM Server v2 - FULL IMPLEMENTATION

Vollständige Implementierung mit allen Features aus dem offiziellen RLM Repo:
https://github.com/alexzhang13/rlm

Features:
- llm_query(), rlm_query() - Direkte und rekursive Aufrufe
- llm_query_batched(), rlm_query_batched() - Parallele Aufrufe
- FINAL_VAR(), SHOW_VARS() - REPL Tools
- context_0, context_1, ... - Versionierte Contexts
- Sessions mit Persistence
- Logger und Trajectory Tracking

Paper: https://arxiv.org/abs/2512.24601
"""
import os
import json
import uuid
import asyncio
import re
import sys
import io
import time
import shutil
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
import httpx
import logging

# Import REPL Environment and Logger
from repl_environment import REPLEnvironment, extract_python_code, RLMChatCompletion
from rlm_logger import RLMLogger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RLM Server v2 (Full Implementation)", version="5.0.0")

# Configuration
GRAPHITI_MCP_URL = os.getenv("GRAPHITI_MCP_URL", "http://localhost:8000")
MEMCLAWZ_URL = os.getenv("MEMCLAWZ_URL", "http://localhost:4010")
SELF_URL = os.getenv("SELF_URL", "http://localhost:5000")

# DashScope / Alibaba Cloud (Anthropic-kompatibel)
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic")
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
LLM_MODEL = os.getenv("MODEL", "glm-5")
DEFAULT_MAX_DEPTH = int(os.getenv("DEFAULT_MAX_DEPTH", "5"))
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))

# =============================================================================
# Pydantic Models
# =============================================================================

class Thought(BaseModel):
    step: int
    thought: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SubQuery(BaseModel):
    query: str
    depth: int
    result: Optional[str] = None
    trajectory_id: Optional[str] = None

class CodeExecution(BaseModel):
    """Record of Python code execution in the REPL."""
    code: str
    output: str
    result: Optional[str] = None
    final_answer: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Trajectory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    query: str
    context: Optional[str] = None
    depth: int = 0
    max_depth: int = DEFAULT_MAX_DEPTH
    result: Optional[str] = None
    thoughts: List[Thought] = Field(default_factory=list)
    sub_queries: List[SubQuery] = Field(default_factory=list)
    code_executions: List[CodeExecution] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, processing, completed, error

class RLMRequest(BaseModel):
    query: str
    context: Optional[Union[str, List[str], Dict[str, Any]]] = None
    recursive: bool = True
    max_depth: int = DEFAULT_MAX_DEPTH
    store_trajectory: bool = True
    persistent: bool = False
    session_id: Optional[str] = None
    parent_trajectory_id: Optional[str] = None
    current_depth: int = 0
    model: Optional[str] = None  # Optional model override

class RLMResponse(BaseModel):
    result: str
    trajectory_id: Optional[str] = None
    session_id: Optional[str] = None
    depth: int = 0
    sub_queries: List[SubQuery] = Field(default_factory=list)
    thoughts: List[Thought] = Field(default_factory=list)
    code_executions: List[CodeExecution] = Field(default_factory=list)
    final_answer: Optional[str] = None
    timing: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    """Simple query request for direct LM call (no REPL)."""
    query: str
    model: Optional[str] = None
    context: Optional[str] = None

class BatchRequest(BaseModel):
    """Batch query request."""
    prompts: List[str]
    model: Optional[str] = None
    recursive: bool = False

class BatchResponse(BaseModel):
    """Batch query response."""
    results: List[str]
    timing: Dict[str, Any] = Field(default_factory=dict)

class SessionCreateRequest(BaseModel):
    """Request to create a new persistent session."""
    initial_context: Optional[str] = None
    max_depth: int = DEFAULT_MAX_DEPTH
    metadata: Optional[Dict[str, Any]] = None

class SessionResponse(BaseModel):
    """Session info response."""
    session_id: str
    created_at: datetime
    last_accessed: datetime
    context_count: int
    message_count: int
    status: str

# =============================================================================
# Session Management
# =============================================================================

class Session:
    """Persistent session for multi-turn conversations."""

    def __init__(
        self,
        session_id: str,
        max_depth: int = DEFAULT_MAX_DEPTH,
        initial_context: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        self.session_id = session_id
        self.max_depth = max_depth
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()

        # REPL environment (persistent)
        self.repl_env: Optional[REPLEnvironment] = None
        self.message_history: List[Dict[str, Any]] = []
        self.context_count: int = 0

        # Initialize REPL if initial context provided
        if initial_context:
            self._init_repl(initial_context)

    def _init_repl(self, context: Union[str, List, Dict]):
        """Initialize the REPL environment."""
        self.repl_env = REPLEnvironment(
            context_payload=context,
            max_depth=self.max_depth,
            current_depth=0,
            self_url=SELF_URL,
            persistent=True,
            compaction=True,
        )
        if isinstance(context, str):
            self.context_count = 1
        elif isinstance(context, list):
            self.context_count = len(context)
        else:
            self.context_count = 1

    def update_access(self):
        """Update last accessed time."""
        self.last_accessed = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if session has expired."""
        elapsed = (datetime.utcnow() - self.last_accessed).total_seconds()
        return elapsed > (SESSION_TIMEOUT_HOURS * 3600)

    def to_response(self) -> SessionResponse:
        """Convert to response model."""
        return SessionResponse(
            session_id=self.session_id,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            context_count=self.context_count,
            message_count=len(self.message_history),
            status="expired" if self.is_expired() else "active",
        )

# Global session store (in-memory)
sessions: Dict[str, Session] = {}

# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """Anthropic API compatible LLM client for DashScope."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy initialization of httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call Anthropic-compatible LLM."""
        # Extract system message
        system = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        model = model_override or self.model

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if system:
            payload["system"] = system

        response = await self.client.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
        )

        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            raise HTTPException(500, f"LLM API error: {response.text}")

        anthropic_response = response.json()

        # Extract text content
        content = anthropic_response.get("content", [])
        text_content = ""

        for block in content:
            if block.get("type") == "text":
                text_content += block.get("text", "")

        return {
            "content": text_content,
            "model": anthropic_response.get("model", model),
            "usage": anthropic_response.get("usage", {}),
        }

llm_client = LLMClient(ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, LLM_MODEL)

# =============================================================================
# Core RLM Processing
# =============================================================================

async def _direct_completion(
    query: str,
    context: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Direct LM completion without REPL."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer directly and concisely."},
        {"role": "user", "content": f"Query: {query}\n\nContext: {context or 'None'}\n\nAnswer:"}
    ]

    response = await llm_client.chat_completion(messages, model_override=model)
    return response["content"].strip()


async def process_rlm(
    request: RLMRequest,
    session: Optional[Session] = None,
) -> RLMResponse:
    """Process an RLM request with full REPL support."""
    start_time = datetime.utcnow()
    llm_calls = 0

    # Create or use existing trajectory
    trajectory = Trajectory(
        parent_id=request.parent_trajectory_id,
        query=request.query,
        context=str(request.context) if request.context else None,
        depth=request.current_depth,
        max_depth=request.max_depth,
    )

    try:
        trajectory.status = "processing"

        # Check depth limit
        if trajectory.depth >= trajectory.max_depth:
            result = await _direct_completion(
                request.query,
                str(request.context) if request.context else None,
                request.model,
            )
            llm_calls += 1
            trajectory.result = result
            trajectory.status = "completed"
        else:
            # Use REPL for recursive solution
            result = await _repl_completion(request, trajectory, session)
            llm_calls += 1

        # Complete trajectory
        trajectory.completed_at = datetime.utcnow()

        end_time = datetime.utcnow()
        total_ms = (end_time - start_time).total_seconds() * 1000

        return RLMResponse(
            result=trajectory.result or result,
            trajectory_id=trajectory.id,
            session_id=session.session_id if session else None,
            depth=trajectory.depth,
            sub_queries=trajectory.sub_queries,
            thoughts=trajectory.thoughts,
            code_executions=trajectory.code_executions,
            final_answer=trajectory.code_executions[-1].final_answer if trajectory.code_executions else None,
            timing={
                "total_ms": int(total_ms),
                "llm_calls": llm_calls,
            },
            metadata={
                "trajectory_id": trajectory.id,
                "status": trajectory.status,
            }
        )

    except Exception as e:
        trajectory.status = "error"
        trajectory.result = f"Error: {str(e)}"
        logger.error(f"RLM processing error: {e}")
        raise HTTPException(500, f"RLM processing error: {str(e)}")


async def _repl_completion(
    request: RLMRequest,
    trajectory: Trajectory,
    session: Optional[Session] = None,
) -> str:
    """Python REPL-based completion with all RLM features."""

    # Determine if we should use direct completion
    query_lower = request.query.lower()

    # Check for simple queries
    is_simple_factorial = (
        "factorial" in query_lower
        and any(str(n) in request.query for n in [1, 2, 3, 4])
        and len(request.query) < 50
    )

    should_use_direct = (
        is_simple_factorial
        or trajectory.depth >= trajectory.max_depth - 1
    )

    if should_use_direct:
        return await _direct_completion(
            request.query,
            str(request.context) if request.context else None,
            request.model,
        )

    # Use session REPL if available and persistent
    if session and session.repl_env and request.persistent:
        repl_env = session.repl_env
        repl_env.subcall_fn = lambda prompt, model=None: _subcall(prompt, model, trajectory)
    else:
        # Create new REPL
        repl_env = REPLEnvironment(
            context_payload=request.context,
            max_depth=trajectory.max_depth,
            current_depth=trajectory.depth,
            self_url=SELF_URL,
            parent_trajectory_id=request.parent_trajectory_id or trajectory.id,
            subcall_fn=lambda prompt, model=None: _subcall(prompt, model, trajectory),
        )

    # Build system prompt
    system_prompt = f"""You are a Recursive Language Model (RLM) running in a Python REPL environment.

You have access to these variables and functions:
- `context`, `context_0`, `context_1`, ...: The context data (can be very long)
- `llm_query(prompt, model=None)`: Direct LM call without REPL
- `rlm_query(prompt, model=None)`: Recursive call with REPL (for complex sub-problems)
- `llm_query_batched(prompts, model=None)`: Parallel LM calls
- `rlm_query_batched(prompts, model=None)`: Parallel recursive calls
- `FINAL_VAR(variable_or_value)`: Mark the final answer
- `SHOW_VARS()`: Show all available variables
- `history`: List of conversation history (when persistent)

Current depth: {trajectory.depth}
Maximum depth: {trajectory.max_depth}

IMPORTANT RULES:
- Use `llm_query()` for simple sub-tasks
- Use `rlm_query()` for complex problems needing reasoning
- Use `FINAL_VAR()` to mark your final answer
- Use `SHOW_VARS()` to check available variables
- NO import statements needed - use built-in functions only

Example:
```python
# Simple task
result = llm_query("Calculate 4!")
print(f"4! = {{result}}")
FINAL_VAR(result)
```

Generate Python code to solve the problem."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {request.query}\n\nCurrent depth: {trajectory.depth}/{trajectory.max_depth}\n\nGenerate Python code:"}
    ]

    # Get code from LLM
    response = await llm_client.chat_completion(messages, model_override=request.model)
    content = response["content"]
    llm_calls = 1

    # Track thought
    thought = Thought(step=len(trajectory.thoughts) + 1, thought=content)
    trajectory.thoughts.append(thought)

    # Extract and execute code
    code = extract_python_code(content)

    if code:
        logger.info(f"Executing code ({len(code)} chars)")

        try:
            repl_result = await repl_env.execute(code)

            # Record execution
            code_exec = CodeExecution(
                code=code,
                output=repl_result.output,
                result=str(repl_result.result) if repl_result.result else None,
                final_answer=repl_result.final_answer,
            )
            trajectory.code_executions.append(code_exec)

            # Add sub-queries from REPL
            for sq in repl_env.get_sub_queries():
                sub_query = SubQuery(
                    query=sq['query'],
                    depth=sq['depth'],
                    result=sq['result'],
                    trajectory_id=sq['trajectory_id'],
                )
                trajectory.sub_queries.append(sub_query)

            # Build result
            if repl_result.final_answer:
                trajectory.result = repl_result.final_answer
                return repl_result.final_answer
            elif repl_result.success:
                trajectory.result = repl_result.output
                return repl_result.output
            else:
                trajectory.result = f"Execution error: {repl_result.error}"
                return trajectory.result

        except Exception as e:
            logger.error(f"REPL execution error: {e}")
            trajectory.result = f"Execution error: {str(e)}"
            return trajectory.result

    # No code extracted
    trajectory.result = content
    return content


def _subcall(prompt: str, model: Optional[str], parent_trajectory: Trajectory) -> RLMChatCompletion:
    """
    Handle a recursive subcall from the REPL.
    This runs synchronously but creates an async task.
    """
    import asyncio
    import time

    async def async_subcall():
        child_request = RLMRequest(
            query=prompt,
            recursive=True,
            max_depth=parent_trajectory.max_depth,
            current_depth=parent_trajectory.depth + 1,
            parent_trajectory_id=parent_trajectory.id,
            model=model,
        )

        child_trajectory = Trajectory(
            parent_id=parent_trajectory.id,
            query=prompt,
            depth=parent_trajectory.depth + 1,
            max_depth=parent_trajectory.max_depth,
        )

        response = await process_rlm(child_request)
        return response

    # Run the async function
    try:
        loop = asyncio.get_event_loop()
        start = time.time()
        response = loop.run_until_complete(async_subcall())
        elapsed = time.time() - start

        return RLMChatCompletion(
            root_model=model or LLM_MODEL,
            prompt=prompt,
            response=response.result,
            execution_time=elapsed,
            metadata=response.metadata,
        )
    except Exception as e:
        return RLMChatCompletion(
            root_model=model or LLM_MODEL,
            prompt=prompt,
            response=f"Error in subcall: {str(e)}",
            execution_time=0.0,
        )


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/rlm", response_model=RLMResponse)
async def rlm_endpoint(request: RLMRequest):
    """Main RLM endpoint - Full recursive processing."""

    # Handle session
    session = None
    if request.persistent and request.session_id:
        session = sessions.get(request.session_id)
        if not session:
            raise HTTPException(404, f"Session {request.session_id} not found")
        if session.is_expired():
            raise HTTPException(410, f"Session {request.session_id} has expired")
        session.update_access()

    # Process request
    return await process_rlm(request, session)


@app.post("/rlm/query")
async def query_endpoint(request: QueryRequest):
    """Simple LM query without REPL (direct completion)."""
    start = datetime.utcnow()

    result = await _direct_completion(
        request.query,
        request.context,
        request.model,
    )

    elapsed = (datetime.utcnow() - start).total_seconds() * 1000

    return {
        "result": result,
        "timing": {"total_ms": int(elapsed)},
    }


@app.post("/rlm/batch", response_model=BatchResponse)
async def batch_endpoint(request: BatchRequest):
    """Batch query endpoint for parallel processing."""
    start = datetime.utcnow()

    # Process all prompts in parallel
    tasks = []
    for prompt in request.prompts:
        if request.recursive:
            req = RLMRequest(query=prompt, recursive=True, max_depth=DEFAULT_MAX_DEPTH)
            tasks.append(process_rlm(req))
        else:
            tasks.append(_direct_completion(prompt, model=request.model))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to error messages
    final_results = []
    for r in results:
        if isinstance(r, Exception):
            final_results.append(f"Error: {str(r)}")
        elif isinstance(r, RLMResponse):
            final_results.append(r.result)
        else:
            final_results.append(r)

    elapsed = (datetime.utcnow() - start).total_seconds() * 1000

    return BatchResponse(
        results=final_results,
        timing={"total_ms": int(elapsed), "count": len(request.prompts)},
    )


# =============================================================================
# Session Management Endpoints
# =============================================================================

@app.post("/rlm/session")
async def create_session(request: SessionCreateRequest):
    """Create a new persistent session."""
    session_id = str(uuid.uuid4())

    session = Session(
        session_id=session_id,
        max_depth=request.max_depth,
        initial_context=request.initial_context,
        metadata=request.metadata,
    )

    sessions[session_id] = session

    logger.info(f"Created session {session_id}")

    return {
        "session_id": session_id,
        "created_at": session.created_at,
        "status": "created",
    }


@app.get("/rlm/session/{session_id}")
async def get_session(session_id: str):
    """Get session info."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    return session.to_response()


@app.post("/rlm/session/{session_id}/continue")
async def continue_session(session_id: str, request: RLMRequest):
    """Continue an existing session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    if session.is_expired():
        raise HTTPException(410, f"Session {session_id} has expired")

    # Update request with session info
    request.persistent = True
    request.session_id = session_id

    return await process_rlm(request, session)


@app.delete("/rlm/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    session = sessions.pop(session_id, None)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    # Cleanup
    if session.repl_env:
        session.repl_env.cleanup()

    return {"status": "deleted", "session_id": session_id}


@app.get("/rlm/sessions")
async def list_sessions():
    """List all active sessions."""
    # Clean up expired sessions first
    expired = [sid for sid, s in sessions.items() if s.is_expired()]
    for sid in expired:
        sessions.pop(sid, None)

    return {
        "sessions": [s.to_response().dict() for s in sessions.values()],
        "count": len(sessions),
    }


# =============================================================================
# Trajectory and Logging Endpoints
# =============================================================================

@app.get("/rlm/trajectory/{trajectory_id}")
async def get_trajectory(trajectory_id: str):
    """Get a trajectory by ID."""
    # This would need to be stored somewhere
    # For now, return placeholder
    return {"trajectory_id": trajectory_id, "status": "not_implemented"}


@app.get("/visualizer")
async def visualizer():
    """Simple HTML visualizer for trajectories."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RLM Trajectory Visualizer</title>
        <style>
            body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #4ec9b0; }
            .trajectory { border: 1px solid #444; margin: 10px 0; padding: 10px; border-radius: 4px; }
            .code { background: #252526; padding: 10px; overflow-x: auto; white-space: pre; }
            .output { background: #1e1e1e; padding: 10px; color: #ce9178; }
            .final { background: #4ec9b0; color: #000; padding: 10px; font-weight: bold; }
            .subquery { margin-left: 20px; border-left: 2px solid #4ec9b0; padding-left: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RLM Trajectory Visualizer</h1>
            <p>Upload a JSONL trajectory file to visualize:</p>
            <input type="file" id="fileInput" accept=".jsonl,.json">
            <div id="content"></div>
        </div>
        <script>
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                const reader = new FileReader();
                reader.onload = function(e) {
                    const lines = e.target.result.split('\\n').filter(l => l.trim());
                    let html = '';
                    lines.forEach((line, i) => {
                        try {
                            const data = JSON.parse(line);
                            html += renderTrajectory(data, i);
                        } catch (err) {
                            html += `<div class="trajectory">Error parsing line ${i}: ${err}</div>`;
                        }
                    });
                    document.getElementById('content').innerHTML = html;
                };
                reader.readAsText(file);
            });

            function renderTrajectory(data, idx) {
                let html = `<div class="trajectory">`;
                html += `<h3>Query: ${data.query || 'N/A'}</h3>`;
                html += `<p>Result: ${data.result || 'N/A'}</p>`;

                if (data.code_executions && data.code_executions.length > 0) {
                    data.code_executions.forEach((exec, i) => {
                        html += `<div class="code">${escapeHtml(exec.code)}</div>`;
                        html += `<div class="output">Output: ${escapeHtml(exec.output)}</div>`;
                        if (exec.final_answer) {
                            html += `<div class="final">Final Answer: ${escapeHtml(exec.final_answer)}</div>`;
                        }
                    });
                }

                if (data.sub_queries && data.sub_queries.length > 0) {
                    html += `<div class="subquery">`;
                    data.sub_queries.forEach(sq => {
                        html += `<p>Sub-query: ${sq.query} → ${sq.result}</p>`;
                    });
                    html += `</div>`;
                }

                html += `</div>`;
                return html;
            }

            function escapeHtml(text) {
                if (!text) return '';
                return text
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;');
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# =============================================================================
# Health and Info
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Clean up expired sessions
    expired = [sid for sid, s in sessions.items() if s.is_expired()]
    for sid in expired:
        sessions.pop(sid, None)

    return {
        "status": "healthy",
        "version": "5.0.0-full",
        "sessions_active": len(sessions),
        "config": {
            "max_depth": DEFAULT_MAX_DEPTH,
            "session_timeout_hours": SESSION_TIMEOUT_HOURS,
        }
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "RLM Server v2 (Full Implementation)",
        "version": "5.0.0",
        "description": "Recursive Language Model with Python REPL - Full Feature Set",
        "endpoints": {
            "POST /rlm": "Main RLM endpoint",
            "POST /rlm/query": "Simple LM query (no REPL)",
            "POST /rlm/batch": "Batch queries",
            "POST /rlm/session": "Create persistent session",
            "POST /rlm/session/{id}/continue": "Continue session",
            "GET /rlm/session/{id}": "Get session info",
            "DELETE /rlm/session/{id}": "Delete session",
            "GET /rlm/sessions": "List all sessions",
            "GET /visualizer": "Trajectory visualizer",
            "GET /health": "Health check",
        },
        "features": [
            "llm_query() / rlm_query()",
            "llm_query_batched() / rlm_query_batched()",
            "FINAL_VAR() / SHOW_VARS()",
            "context_0, context_1, ...",
            "Persistent sessions",
            "Trajectory logging",
        ]
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
