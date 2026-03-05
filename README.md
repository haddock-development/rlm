# RLM Server v2 - Recursive Language Models

> **REST API Implementation of Recursive Language Models (RLMs)**
> Based on the official RLM repository: https://github.com/alexzhang13/rlm
> Paper: [Recursive Language Models (Zhang et al., 2025/2026)](https://arxiv.org/abs/2512.24601)

[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)

## Overview

This repository provides a **production-ready REST API implementation** of Recursive Language Models (RLMs), enabling language models to handle near-infinite length contexts by allowing the LM to *programmatically* examine, decompose, and recursively call itself over its input.

### What makes RLMs different?

Traditional LLM calls:
```python
response = llm.completion(prompt, model)  # Single-shot
```

RLM calls with REPL environment:
```python
response = rlm.completion(prompt, model)  # Can spawn sub-calls, execute code, iterate
```

RLMs offload context as variables in a REPL environment that the LM can interact with and launch sub-LM calls inside of.

## Features

### Core REPL Tools (100% Compatible with Original)

| Tool | Description | Status |
|------|-------------|--------|
| `FINAL_VAR(value)` | Mark a variable/value as the final answer | ✅ |
| `SHOW_VARS()` | Display all available REPL variables | ✅ |
| `llm_query(prompt, model=None)` | Direct LM call without REPL | ✅ |
| `rlm_query(prompt, model=None)` | Recursive RLM call with REPL | ✅ |
| `llm_query_batched(prompts, model=None)` | Parallel LM calls | ✅ |
| `rlm_query_batched(prompts, model=None)` | Parallel recursive calls | ✅ |

### Context Management

| Feature | Description | Status |
|---------|-------------|--------|
| `context` / `context_0` | Primary context variable | ✅ |
| `context_1`, `context_2`, ... | Versioned context variables | ✅ |
| `load_context(payload)` | Load context into environment | ✅ |
| `add_context(payload, index=None)` | Add context with auto/manual index | ✅ |
| `get_context_count()` | Get number of loaded contexts | ✅ |

### History & Persistence

| Feature | Description | Status |
|---------|-------------|--------|
| `history` / `history_0` | Conversation history variable | ✅ |
| `add_history(messages, index=None)` | Store message history | ✅ |
| `get_history_count()` | Get number of histories | ✅ |
| `compaction` | History compression support | ✅ |
| `append_compaction_entry(entry)` | Add to compaction history | ✅ |
| Persistent Sessions | Multi-turn conversations | ✅ |

### Safety & Sandboxing

| Feature | Implementation |
|---------|----------------|
| Blocked builtins | `eval`, `exec`, `input`, `compile`, `globals`, `locals` |
| Safe builtins | 134 allowed functions/types |
| Execution timeout | 30 seconds default |
| Reserved names | Protected tool names cannot be overridden |
| Custom tools validation | Prevents override of reserved names |

## Quick Start

### Prerequisites

- Docker Desktop or Docker Engine
- Anthropic-compatible API key (or OpenAI, etc.)

### 1. Clone and Build

```bash
git clone <your-repo-url>
cd rlm_v2

# Build Docker image
docker build -t rlm-v2:latest .
```

### 2. Configure Environment

Create environment variables or export directly:

```bash
# For Alibaba Cloud (DashScope) - Anthropic-compatible
export ANTHROPIC_BASE_URL="https://coding-intl.dashscope.aliyuncs.com/apps/anthropic"
export ANTHROPIC_AUTH_TOKEN="your-token-here"
export MODEL="glm-5"

# For OpenAI
export ANTHROPIC_BASE_URL="https://api.openai.com/v1"
export ANTHROPIC_AUTH_TOKEN="sk-..."
export MODEL="gpt-4"

# Optional settings
export DEFAULT_MAX_DEPTH="5"
export SESSION_TIMEOUT_HOURS="24"
export SELF_URL="http://localhost:5000"
```

### 3. Run Container

```bash
docker run -d -p 5000:5000 \
  -e ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL}" \
  -e ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN}" \
  -e MODEL="${MODEL}" \
  -e SELF_URL="http://localhost:5000" \
  --name rlm-v2-server \
  rlm-v2:latest
```

### 4. Verify Installation

```bash
# Health check
curl http://localhost:5000/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "5.0.0-full",
#   "sessions_active": 0,
#   "config": {
#     "max_depth": 5,
#     "session_timeout_hours": 24
#   }
# }
```

## API Reference

### Core Endpoints

#### `POST /rlm` - Main RLM Completion

Execute a recursive language model completion with full REPL support.

```bash
curl -X POST http://localhost:5000/rlm \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Calculate 15 * 23 using Python",
    "context": "You are a math assistant. Use Python for calculations.",
    "recursive": true,
    "max_depth": 3,
    "max_iterations": 5,
    "store_trajectory": true,
    "persistent": false,
    "session_id": null,
    "model": null
  }'
```

**Response:**
```json
{
  "result": "345",
  "trajectory_id": "uuid-here",
  "session_id": null,
  "depth": 0,
  "thoughts": [...],
  "code_executions": [...],
  "final_answer": "345",
  "timing": {
    "total_ms": 5817,
    "llm_calls": 1
  },
  "metadata": {...}
}
```

#### `POST /rlm/query` - Direct LM Query

Simple LM completion without REPL or recursion.

```bash
curl -X POST http://localhost:5000/rlm/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "context": "Answer briefly.",
    "model": null
  }'
```

**Response:**
```json
{
  "result": "Paris",
  "timing": {
    "total_ms": 1234
  }
}
```

#### `POST /rlm/batch` - Batch Processing

Process multiple prompts in parallel.

```bash
curl -X POST http://localhost:5000/rlm/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["1+1=?", "2+2=?", "3+3=?"],
    "model": null,
    "recursive": false
  }'
```

**Response:**
```json
{
  "results": ["2", "4", "6"],
  "timing": {
    "total_ms": 2928,
    "count": 3
  }
}
```

### Session Management

#### `POST /rlm/session` - Create Session

Create a persistent session for multi-turn conversations.

```bash
curl -X POST http://localhost:5000/rlm/session \
  -H "Content-Type: application/json" \
  -d '{
    "initial_context": "Context for this conversation",
    "max_depth": 5,
    "metadata": {"user_id": "123"}
  }'
```

**Response:**
```json
{
  "session_id": "uuid-here",
  "created_at": "2026-03-05T09:11:57.334203",
  "status": "created"
}
```

#### `POST /rlm/session/{id}/continue` - Continue Session

Continue an existing session.

```bash
curl -X POST http://localhost:5000/rlm/session/{session_id}/continue \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was my initial context?",
    "max_depth": 2,
    "max_iterations": 5
  }'
```

#### `GET /rlm/sessions` - List Sessions

```bash
curl http://localhost:5000/rlm/sessions
```

#### `DELETE /rlm/session/{id}` - Delete Session

```bash
curl -X DELETE http://localhost:5000/rlm/session/{session_id}
```

### Utility Endpoints

#### `GET /health` - Health Check

```bash
curl http://localhost:5000/health
```

## Usage Examples

### Example 1: Simple Calculation with Python

```bash
curl -X POST http://localhost:5000/rlm \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Calculate 15 * 23 using Python. Store result in variable and use FINAL_VAR.",
    "context": "Use Python code blocks.",
    "max_depth": 2
  }'
```

The LLM might respond with:
```python
result = 15 * 23
print(f"15 * 23 = {result}")
FINAL_VAR(result)
```

**Result:** `345`

### Example 2: Multi-Context Processing

```bash
# First query with context
curl -X POST http://localhost:5000/rlm \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze context_0 and context_1, then compare them.",
    "context": ["First document content...", "Second document content..."],
    "max_depth": 3
  }'
```

In the REPL:
```python
# Access versioned contexts
first = context_0
second = context_1

# Compare
combined = f"First: {len(first)} chars, Second: {len(second)} chars"
FINAL_VAR(combined)
```

### Example 3: Persistent Session

```bash
# Create session
SESSION_ID=$(curl -s -X POST http://localhost:5000/rlm/session \
  -H "Content-Type: application/json" \
  -d '{"initial_context": "We are discussing Python programming."}' \
  | jq -r '.session_id')

# First message
curl -X POST http://localhost:5000/rlm/session/${SESSION_ID}/continue \
  -H "Content-Type: application/json" \
  -d '{"query": "What is my context?"}'

# Continue conversation (history is preserved)
curl -X POST http://localhost:5000/rlm/session/${SESSION_ID}/continue \
  -H "Content-Type: application/json" \
  -d '{"query": "Show all variables with SHOW_VARS()"}'
```

### Example 4: Batched Queries

```bash
curl -X POST http://localhost:5000/rlm/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "Translate 'hello' to French",
      "Translate 'hello' to Spanish",
      "Translate 'hello' to German"
    ],
    "recursive": false
  }'
```

## Architecture

### Components

| File | Purpose |
|------|---------|
| `rlm_server_v2.py` | FastAPI server with all endpoints |
| `repl_environment.py` | Full REPL implementation with all tools |
| `safe_builtins.py` | Safety configuration (blocked/allowed builtins) |
| `rlm_logger.py` | Trajectory logging and visualization support |
| `Dockerfile` | Production container configuration |

### REPL Environment Features

```python
class REPLEnvironment:
    # Core namespace
    globals: Dict[str, Any]  # Builtins and tools
    locals: Dict[str, Any]   # User-defined variables

    # Tools available to LLM
    FINAL_VAR(value)          # Mark final answer
    SHOW_VARS()               # Show all variables
    llm_query(prompt)         # Direct LM call
    rlm_query(prompt)         # Recursive RLM call
    llm_query_batched(prompts)  # Parallel LM calls
    rlm_query_batched(prompts)  # Parallel RLM calls

    # Context access
    context, context_0, context_1, ...

    # History access (in persistent sessions)
    history, history_0, history_1, ...
```

### Security Model

```python
# Blocked builtins (set to None)
BLOCKED = {
    "eval": None,
    "exec": None,
    "input": None,
    "compile": None,
    "globals": None,
    "locals": None,
    "__import__": None,
}

# Allowed builtins (134 functions/types)
SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    # ... and 131 more
}
```

## Differences from Original RLM

| Aspect | Original (alexzhang13/rlm) | This Implementation |
|--------|---------------------------|---------------------|
| **Interface** | Python library | REST API |
| **Installation** | `pip install rlms` | Docker container |
| **Usage** | `from rlm import RLM` | HTTP API calls |
| **Environments** | local, docker, modal, prime, e2b | Local Docker only |
| **Scalability** | Single process | Containerized, stateless |
| **Persistence** | In-memory | Sessions with timeout |

## Development

### Local Development (without Docker)

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server locally
python rlm_server_v2.py
```

### Running Tests

```bash
# Health check
curl http://localhost:5000/health

# Simple query
curl -X POST http://localhost:5000/rlm/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Say hi"}'

# Full RLM
curl -X POST http://localhost:5000/rlm \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate 2+2 with Python", "max_depth": 2}'
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_BASE_URL` | Required | API base URL |
| `ANTHROPIC_AUTH_TOKEN` | Required | API authentication token |
| `MODEL` | `glm-5` | Default model name |
| `DEFAULT_MAX_DEPTH` | `5` | Maximum recursion depth |
| `SESSION_TIMEOUT_HOURS` | `24` | Session expiration time |
| `SELF_URL` | `http://localhost:5000` | Self-referential URL |
| `GRAPHITI_MCP_URL` | `http://localhost:8000` | Graphiti MCP endpoint |
| `MEMCLAWZ_URL` | `http://localhost:4010` | Vector memory endpoint |

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs rlm-v2-server

# Verify environment variables
echo $ANTHROPIC_AUTH_TOKEN

# Rebuild and restart
docker build -t rlm-v2:latest .
docker stop rlm-v2-server && docker rm rlm-v2-server
docker run -d -p 5000:5000 -e ... rlm-v2:latest
```

### API returns errors

```bash
# Check if server is running
curl http://localhost:5000/health

# Test direct query
curl -X POST http://localhost:5000/rlm/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

### Rate limiting

If you see `Weekly/Monthly Limit Exhausted` errors:
- Check your API quota
- Wait for quota reset
- Use a different API key

## References

- **Original RLM Repository**: https://github.com/alexzhang13/rlm
- **Paper**: [Recursive Language Models (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)
- **Documentation**: https://alexzhang13.github.io/rlm/
- **Blog Post**: https://alexzhang13.github.io/blog/2025/rlm/

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@misc{zhang2026recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2026},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```

## License

This implementation is based on the RLM paper and original repository by Zhang et al. Please refer to the original repository for licensing information.

---

**Maintained by**: haddock-development
**Status**: Production Ready
**Version**: 5.0.0-full
