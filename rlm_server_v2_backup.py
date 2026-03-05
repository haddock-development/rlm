"""RLM Server v2 - Echtes rekursives Language Model System mit Trajectory-Speicher."""
import os
import json
import uuid
import asyncio
import re
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RLM Server v2 (True Recursive)", version="3.0.0")

# Configuration
GRAPHITI_MCP_URL = os.getenv("GRAPHITI_MCP_URL", "http://localhost:8000")
MEMCLAWZ_URL = os.getenv("MEMCLAWZ_URL", "http://localhost:4010")
SELF_URL = os.getenv("SELF_URL", "http://localhost:5000")

# DashScope / Alibaba Cloud (Anthropic-kompatibel)
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic")
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
LLM_MODEL = os.getenv("MODEL", "kimi-k2.5")
DEFAULT_MAX_DEPTH = int(os.getenv("DEFAULT_MAX_DEPTH", "5"))

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

class ToolCall(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None
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
    tool_calls: List[ToolCall] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, processing, completed, error

class RLMRequest(BaseModel):
    query: str
    context: Optional[str] = None
    recursive: bool = True
    max_depth: int = DEFAULT_MAX_DEPTH
    store_trajectory: bool = True
    tools: List[str] = Field(default_factory=lambda: ["rlm_call", "file_read", "search_memclawz"])
    parent_trajectory_id: Optional[str] = None
    current_depth: int = 0

class RLMResponse(BaseModel):
    result: str
    trajectory_id: Optional[str] = None
    depth: int = 0
    sub_queries: List[SubQuery] = Field(default_factory=list)
    thoughts: List[Thought] = Field(default_factory=list)
    timing: Dict[str, Any] = Field(default_factory=dict)

class TrajectoryTree(BaseModel):
    trajectory: Trajectory
    children: List["TrajectoryTree"] = Field(default_factory=list)

# =============================================================================
# Graphiti MCP Integration with SSE
# =============================================================================

class GraphitiMCPClient:
    """Client for Graphiti MCP Server using SSE protocol."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.message_counter = 0

    def _get_next_id(self) -> int:
        self.message_counter += 1
        return self.message_counter

    async def call_tool(self, tool_name: str, params: Dict) -> Optional[Dict]:
        """Call a Graphiti MCP tool via POST endpoint."""
        try:
            # Graphiti MCP uses /sse endpoint for initialization
            # But we can try POST JSON-RPC style
            json_rpc_request = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": f"tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params
                }
            }

            response = await self.client.post(
                f"{self.base_url}/messages",
                json=json_rpc_request,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    return result["result"]
                elif "error" in result:
                    logger.warning(f"MCP error: {result['error']}")
                    return None

            # Fallback: Try direct tool endpoint
            response = await self.client.post(
                f"{self.base_url}/tools/{tool_name}",
                json=params,
                timeout=30.0
            )

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.debug(f"MCP call failed: {e}")
            return None

    async def add_episode(self, name: str, episode_body: str, source: str = "text", source_description: str = "") -> Optional[str]:
        """Add an episode to Graphiti."""
        result = await self.call_tool("add_episode", {
            "name": name,
            "episode_body": episode_body,
            "source": source,
            "source_description": source_description
        })
        return result.get("uuid") if result else None

    async def search_nodes(self, query: str, max_nodes: int = 10, entity: Optional[str] = None) -> List[Dict]:
        """Search for nodes in Graphiti."""
        params = {"query": query, "max_nodes": max_nodes}
        if entity:
            params["entity"] = entity

        result = await self.call_tool("search_nodes", params)
        return result if result else []

    async def search_facts(self, query: str, max_facts: int = 10) -> List[Dict]:
        """Search for facts in Graphiti."""
        result = await self.call_tool("search_facts", {
            "query": query,
            "max_facts": max_facts
        })
        return result if result else []


class GraphitiStore:
    """Trajectory storage via Graphiti MCP with local fallback."""

    def __init__(self, base_url: str):
        self.mcp_client = GraphitiMCPClient(base_url)
        self.local_storage: Dict[str, Dict] = {}
        self._available = None

    async def _check_availability(self) -> bool:
        """Check if Graphiti MCP is available."""
        if self._available is None:
            try:
                # Try a simple search to check connectivity
                result = await self.mcp_client.search_nodes("test", max_nodes=1)
                self._available = True
                logger.info("Graphiti MCP connection established")
            except Exception as e:
                self._available = False
                logger.warning(f"Graphiti MCP not available: {e}")
        return self._available

    async def add_trajectory(self, trajectory: Trajectory) -> str:
        """Store a trajectory in Graphiti or locally."""
        # Always store locally as fallback
        self.local_storage[trajectory.id] = trajectory.model_dump()

        # Try Graphiti if available
        if await self._check_availability():
            try:
                uuid = await self.mcp_client.add_episode(
                    name=f"Trajectory-{trajectory.id[:8]}",
                    episode_body=json.dumps({
                        "entity_type": "Trajectory",
                        "id": trajectory.id,
                        "query": trajectory.query,
                        "context": trajectory.context,
                        "depth": trajectory.depth,
                        "max_depth": trajectory.max_depth,
                        "status": trajectory.status,
                        "parent_id": trajectory.parent_id,
                        "created_at": trajectory.created_at.isoformat(),
                    }),
                    source="json",
                    source_description="RLM Trajectory"
                )

                if uuid:
                    logger.info(f"Stored trajectory {trajectory.id} in Graphiti with UUID {uuid}")
                else:
                    logger.info(f"Stored trajectory {trajectory.id} locally (Graphiti no UUID)")
            except Exception as e:
                logger.warning(f"Failed to store in Graphiti: {e}, using local storage")
        else:
            logger.info(f"Stored trajectory {trajectory.id} locally (Graphiti unavailable)")

        return trajectory.id

    async def add_thought(self, trajectory_id: str, thought: Thought) -> None:
        """Store a thought."""
        if trajectory_id in self.local_storage:
            if "thoughts" not in self.local_storage[trajectory_id]:
                self.local_storage[trajectory_id]["thoughts"] = []
            self.local_storage[trajectory_id]["thoughts"].append(thought.model_dump())

        if await self._check_availability():
            try:
                await self.mcp_client.add_episode(
                    name=f"Thought-{trajectory_id[:8]}-{thought.step}",
                    episode_body=json.dumps({
                        "entity_type": "Thought",
                        "step": thought.step,
                        "content": thought.thought,
                        "trajectory_id": trajectory_id,
                        "timestamp": thought.timestamp.isoformat(),
                    }),
                    source="json",
                    source_description="RLM Thought"
                )
            except Exception as e:
                logger.debug(f"Failed to store thought in Graphiti: {e}")

    async def add_subquery(self, parent_id: str, subquery: SubQuery) -> None:
        """Store a subquery."""
        if parent_id in self.local_storage:
            if "sub_queries" not in self.local_storage[parent_id]:
                self.local_storage[parent_id]["sub_queries"] = []
            self.local_storage[parent_id]["sub_queries"].append(subquery.model_dump())

        if await self._check_availability():
            try:
                await self.mcp_client.add_episode(
                    name=f"SubQuery-{parent_id[:8]}",
                    episode_body=json.dumps({
                        "entity_type": "SubQuery",
                        "query": subquery.query,
                        "depth": subquery.depth,
                        "result": subquery.result,
                        "parent_trajectory_id": parent_id,
                        "trajectory_id": subquery.trajectory_id,
                    }),
                    source="json",
                    source_description="RLM SubQuery"
                )
            except Exception as e:
                logger.debug(f"Failed to store subquery in Graphiti: {e}")

    async def update_trajectory_result(self, trajectory_id: str, result: str, status: str = "completed") -> None:
        """Update trajectory with final result."""
        if trajectory_id in self.local_storage:
            self.local_storage[trajectory_id]["result"] = result
            self.local_storage[trajectory_id]["status"] = status

        if await self._check_availability():
            try:
                await self.mcp_client.add_episode(
                    name=f"Trajectory-Result-{trajectory_id[:8]}",
                    episode_body=json.dumps({
                        "entity_type": "Trajectory",
                        "id": trajectory_id,
                        "result": result,
                        "status": status,
                        "completed_at": datetime.utcnow().isoformat(),
                    }),
                    source="json",
                    source_description="RLM Trajectory Update"
                )
            except Exception as e:
                logger.debug(f"Failed to update trajectory in Graphiti: {e}")

    async def search_trajectories(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for similar trajectories."""
        if await self._check_availability():
            try:
                results = await self.mcp_client.search_nodes(
                    query=query,
                    max_nodes=max_results,
                    entity="Trajectory"
                )
                if results:
                    return results
            except Exception as e:
                logger.debug(f"Graphiti search failed: {e}")

        # Local search fallback
        results = []
        for traj_id, traj in self.local_storage.items():
            if query.lower() in str(traj.get("query", "")).lower():
                results.append(traj)
        return results[:max_results]

# Initialize Graphiti store
graphiti_store = GraphitiStore(GRAPHITI_MCP_URL)

# =============================================================================
# Tool System
# =============================================================================

class ToolRegistry:
    """Registry for RLM tools."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, name: str):
        """Decorator to register a tool."""
        def decorator(func: Callable):
            self.tools[name] = func
            return func
        return decorator

    async def execute(self, name: str, params: Dict[str, Any]) -> Any:
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return await self.tools[name](**params)

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Get tool descriptions for LLM (Anthropic format)."""
        return [
            {
                "name": "rlm_call",
                "description": "Make a recursive RLM call for sub-problems. Use when the problem can be broken down into smaller sub-problems.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The sub-problem to solve"},
                        "context": {"type": "string", "description": "Optional context for the sub-problem"},
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "file_read",
                "description": "Read a file from the filesystem",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "file_write",
                "description": "Write content to a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "search_memclawz",
                "description": "Search documents in memclawz vector store",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_web",
                "description": "Search the web for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"]
                }
            }
        ]

tool_registry = ToolRegistry()

# Tool implementations

@tool_registry.register("rlm_call")
async def tool_rlm_call(query: str, context: str = "", trajectory_id: str = "", depth: int = 0) -> str:
    """Make a recursive RLM call."""
    # This will be called internally by the RLM processor
    # The actual implementation is in the process_rlm function
    raise NotImplementedError("rlm_call is handled internally by the RLM processor")

@tool_registry.register("file_read")
async def tool_file_read(path: str) -> str:
    """Read a file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool_registry.register("file_write")
async def tool_file_write(path: str, content: str) -> str:
    """Write a file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool_registry.register("search_memclawz")
async def tool_search_memclawz(query: str, top_k: int = 5) -> List[Dict]:
    """Search memclawz vector store."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEMCLAWZ_URL}/search",
                json={"query": query, "top_k": top_k},
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json().get("results", [])
            return [{"error": f"Search failed: {response.status_code}"}]
    except Exception as e:
        return [{"error": str(e)}]

@tool_registry.register("search_web")
async def tool_search_web(query: str) -> str:
    """Search the web (placeholder - requires Exa MCP)."""
    return f"Web search for '{query}' - implement with Exa MCP if needed"

# =============================================================================
# LLM Integration
# =============================================================================

class LLMClient:
    """Anthropic API compatible LLM client for DashScope."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Call Anthropic-compatible LLM with optional tool support."""
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

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if system:
            payload["system"] = system

        # Convert tools to Anthropic format
        if tools:
            anthropic_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    anthropic_tools.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "input_schema": tool["function"]["parameters"]
                    })
            payload["tools"] = anthropic_tools

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

        # Convert Anthropic response to OpenAI-like format
        content = anthropic_response.get("content", [])
        text_content = ""
        tool_calls = []

        for block in content:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": text_content if text_content else None,
                    "tool_calls": tool_calls if tool_calls else None
                }
            }]
        }

llm_client = LLMClient(ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, LLM_MODEL)

# =============================================================================
# Core RLM Processor
# =============================================================================

class RLMProcessor:
    """Main RLM processing logic with recursion support."""

    def __init__(self):
        self.active_trajectories: Dict[str, Trajectory] = {}

    async def process(
        self,
        request: RLMRequest,
        trajectory: Trajectory,
    ) -> RLMResponse:
        """Process an RLM request with potential recursion."""
        start_time = datetime.utcnow()
        llm_calls = 0

        try:
            # Store trajectory in memory
            self.active_trajectories[trajectory.id] = trajectory
            trajectory.status = "processing"

            # Store in Graphiti if enabled
            if request.store_trajectory:
                await graphiti_store.add_trajectory(trajectory)

            # Check depth limit
            if trajectory.depth >= trajectory.max_depth:
                result = await self._direct_completion(request, trajectory)
                llm_calls += 1
            else:
                # Attempt recursive solution with tool calling
                result = await self._recursive_completion(request, trajectory)
                llm_calls += 1  # At least one call

            # Complete trajectory
            trajectory.result = result
            trajectory.status = "completed"
            trajectory.completed_at = datetime.utcnow()

            # Update Graphiti with result
            if request.store_trajectory:
                await graphiti_store.update_trajectory_result(
                    trajectory.id, result, "completed"
                )

            end_time = datetime.utcnow()
            total_ms = (end_time - start_time).total_seconds() * 1000

            return RLMResponse(
                result=result,
                trajectory_id=trajectory.id,
                depth=trajectory.depth,
                sub_queries=trajectory.sub_queries,
                thoughts=trajectory.thoughts,
                timing={
                    "total_ms": int(total_ms),
                    "llm_calls": llm_calls,
                }
            )

        except Exception as e:
            trajectory.status = "error"
            trajectory.result = f"Error: {str(e)}"
            logger.error(f"RLM processing error: {e}")
            raise HTTPException(500, f"RLM processing error: {str(e)}")

    async def _direct_completion(self, request: RLMRequest, trajectory: Trajectory) -> str:
        """Direct LLM completion without recursion."""
        system_prompt = """You are a helpful assistant. Answer the user's query directly and concisely."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_prompt(request)}
        ]

        response = await llm_client.chat_completion(messages)
        return response["choices"][0]["message"]["content"]

    async def _recursive_completion(self, request: RLMRequest, trajectory: Trajectory) -> str:
        """Recursive completion with tool calling support."""

        system_prompt = f"""You are a Recursive Language Model (RLM). You can solve complex problems by:
1. Breaking them into smaller sub-problems and using the `rlm_call` tool
2. Reading files with `file_read`
3. Searching documents with `search_memclawz`
4. Searching the web with `search_web`

Current depth: {trajectory.depth}
Maximum depth: {trajectory.max_depth}

When you need to break down a problem:
- Analyze what sub-problems need solving
- Use `rlm_call` for each sub-problem
- Synthesize the results into your final answer

To use tools, output them in this XML format:
<tool_call name="rlm_call">{{"query": "sub-problem", "context": "optional"}}</tool_call>
<tool_call name="file_read">{{"path": "/path/to/file"}}</tool_call>

Think step by step and explain your reasoning."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_prompt(request)}
        ]

        # Get available tools
        tools = self._build_tools_for_llm(request.tools)

        # First LLM call - may request tool usage
        response = await llm_client.chat_completion(messages, tools=tools)
        message = response["choices"][0]["message"]
        content = message.get("content", "")

        # Track the thought
        thought = Thought(step=len(trajectory.thoughts) + 1, thought=content)
        trajectory.thoughts.append(thought)
        if request.store_trajectory:
            await graphiti_store.add_thought(trajectory.id, thought)

        # Check for tool calls (structured API or XML parsing)
        tool_calls = message.get("tool_calls", [])

        # If no structured tool calls, try XML parsing
        if not tool_calls:
            tool_calls = self._parse_xml_tool_calls(content)

        if tool_calls:
            # Add assistant message
            messages.append({"role": "assistant", "content": content})

            # Process each tool call
            for i, tool_call in enumerate(tool_calls):
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                # Record tool call
                tc = ToolCall(tool_name=function_name, parameters=function_args)
                trajectory.tool_calls.append(tc)

                # Execute tool
                if function_name == "rlm_call":
                    # Handle recursive RLM call
                    sub_result = await self._execute_recursive_call(
                        function_args, request, trajectory
                    )
                    tc.result = sub_result
                else:
                    # Execute other tools
                    try:
                        result = await tool_registry.execute(function_name, function_args)
                        tc.result = result
                    except Exception as e:
                        tc.result = f"Error: {str(e)}"

                # Add tool response to messages (as user message for Anthropic compatibility)
                messages.append({
                    "role": "user",
                    "content": f"Tool '{function_name}' result: {str(tc.result) if tc.result else 'No result'}",
                })

            # Final completion with tool results
            final_response = await llm_client.chat_completion(messages)
            return final_response["choices"][0]["message"]["content"]

        # No tool calls - direct response
        return content

    def _parse_xml_tool_calls(self, content: str) -> List[Dict]:
        """Parse XML-style tool calls from LLM content."""
        tool_calls = []
        import re

        logger.info(f"Parsing content for tool calls: {content[:200]}...")

        # Pattern 1: <tool_call name="rlm_call">{...}</tool_call> (standard)
        pattern1 = r'<tool_call\s+name="([^"]+)">(.*?)</tool_call>'
        matches1 = re.findall(pattern1, content, re.DOTALL)
        logger.info(f"Pattern 1 matches: {len(matches1)}")
        for name, args_str in matches1:
            try:
                args_str = args_str.strip()
                args_json = json.loads(args_str)
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str
                    }
                })
                logger.info(f"Parsed tool call: {name}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call args: {args_str}")

        # Pattern 2: <tool_call>rlm_call">{...}</tool_call> (anthropic function format)
        pattern2 = r'<tool_call>(\w+)">(.*?)<\/tool_call>'
        matches2 = re.findall(pattern2, content, re.DOTALL)
        logger.info(f"Pattern 2 matches: {len(matches2)}")
        for name, args_str in matches2:
            try:
                args_str = args_str.strip()
                args_json = json.loads(args_str)
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str
                    }
                })
                logger.info(f"Parsed tool call (pattern 2): {name}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call args (pattern 2): {args_str}")

        # Pattern 3: <tool_call name="rlm_call">{...}</arg_value>tool_call> (malformed)
        pattern3 = r'<tool_call\s+name="([^"]+)">(.*?)<\/[^<]*?tool_call>'
        matches3 = re.findall(pattern3, content, re.DOTALL)
        logger.info(f"Pattern 3 matches: {len(matches3)}")
        for name, args_str in matches3:
            # Skip duplicates
            if any(tc["function"]["name"] == name for tc in tool_calls):
                continue
            try:
                args_str = args_str.strip()
                args_json = json.loads(args_str)
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str
                    }
                })
                logger.info(f"Parsed malformed tool call: {name}")
            except json.JSONDecodeError:
                pass

        logger.info(f"Total tool calls parsed: {len(tool_calls)}")
        return tool_calls

    async def _execute_recursive_call(
        self,
        args: Dict[str, Any],
        parent_request: RLMRequest,
        parent_trajectory: Trajectory,
    ) -> str:
        """Execute a recursive RLM call."""
        # Create sub-query record
        sub_query = SubQuery(
            query=args.get("query", ""),
            depth=parent_trajectory.depth + 1,
        )
        parent_trajectory.sub_queries.append(sub_query)

        # Create child trajectory
        child_trajectory = Trajectory(
            parent_id=parent_trajectory.id,
            query=args.get("query", ""),
            context=args.get("context", ""),
            depth=parent_trajectory.depth + 1,
            max_depth=parent_trajectory.max_depth,
        )

        # Store subquery in Graphiti
        if parent_request.store_trajectory:
            await graphiti_store.add_subquery(parent_trajectory.id, sub_query)

        # Make recursive call via HTTP to self
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{SELF_URL}/rlm",
                    json={
                        "query": args.get("query", ""),
                        "context": args.get("context", ""),
                        "recursive": True,
                        "max_depth": parent_trajectory.max_depth,
                        "store_trajectory": parent_request.store_trajectory,
                        "tools": parent_request.tools,
                        "parent_trajectory_id": parent_trajectory.id,
                        "current_depth": parent_trajectory.depth + 1,
                    },
                    timeout=300.0,
                )

                if response.status_code == 200:
                    result_data = response.json()
                    sub_query.result = result_data.get("result", "")
                    sub_query.trajectory_id = result_data.get("trajectory_id", "")
                    return sub_query.result
                else:
                    error_msg = f"Recursive call failed: {response.text}"
                    sub_query.result = error_msg
                    return error_msg
            except Exception as e:
                error_msg = f"Recursive call error: {str(e)}"
                sub_query.result = error_msg
                return error_msg

    def _build_prompt(self, request: RLMRequest) -> str:
        """Build the prompt from request."""
        if request.context:
            return f"Context: {request.context}\n\nQuery: {request.query}"
        return request.query

    def _build_tools_for_llm(self, tool_names: List[str]) -> List[Dict]:
        """Build tool definitions for LLM (Anthropic format)."""
        all_tools = tool_registry.get_tool_descriptions()
        return [t for t in all_tools if t["name"] in tool_names]

# Initialize processor
rlm_processor = RLMProcessor()

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "3.0.0-true-recursive",
        "model": LLM_MODEL,
        "graphiti_url": GRAPHITI_MCP_URL,
        "memclawz_url": MEMCLAWZ_URL,
        "self_url": SELF_URL,
    }

@app.post("/rlm", response_model=RLMResponse)
async def rlm_endpoint(request: RLMRequest):
    """
    Main RLM endpoint with recursive capabilities.

    The LLM can:
    - Answer directly
    - Break problems into sub-problems via rlm_call tool
    - Use other tools (file_read, search_memclawz, etc.)

    All interactions are stored as trajectories in Graphiti.
    """
    # Create trajectory
    trajectory = Trajectory(
        query=request.query,
        context=request.context,
        depth=request.current_depth,
        max_depth=request.max_depth,
        parent_id=request.parent_trajectory_id,
    )

    # Process request
    return await rlm_processor.process(request, trajectory)

@app.get("/trajectory/{trajectory_id}")
async def get_trajectory(trajectory_id: str):
    """Get a trajectory by ID."""
    if trajectory_id in rlm_processor.active_trajectories:
        return rlm_processor.active_trajectories[trajectory_id]

    # Try to fetch from Graphiti
    results = await graphiti_store.search_trajectories(f"id:{trajectory_id}", max_results=1)
    if results:
        return results[0]

    raise HTTPException(404, "Trajectory not found")

@app.get("/trajectories/search")
async def search_trajectories(query: str, max_results: int = 10):
    """Search for trajectories."""
    results = await graphiti_store.search_trajectories(query, max_results)
    return {"results": results}

@app.get("/tools")
async def list_tools():
    """List available tools."""
    return {"tools": tool_registry.get_tool_descriptions()}

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
