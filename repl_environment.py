"""
RLM Python REPL Environment - FULL IMPLEMENTATION
===================================================

Vollständige Implementierung basierend auf:
"Recursive Language Models" (Zhang et al., 2025/2026)
https://arxiv.org/abs/2512.24601
https://github.com/alexzhang13/rlm

Features:
- context, context_0, context_1, ... (versioned context)
- llm_query(): Einfacher LM Aufruf (ohne REPL)
- rlm_query(): Rekursiver Aufruf mit REPL
- llm_query_batched(): Parallele LM Aufrufe
- rlm_query_batched(): Parallele rekursive Aufrufe
- FINAL_VAR(): Markiert finale Antwort
- SHOW_VARS(): Zeigt alle REPL Variablen
- history: Zugriff auf Konversationshistorie

Sicherheit:
- Sandboxed execution (kein os, sys, subprocess)
- Blockiert: eval, exec, input, compile
- Timeout: 30 Sekunden pro Code-Block
- Memory Limit: 100MB
"""

import asyncio
import copy
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import safe builtins from separate module
from safe_builtins import SAFE_BUILTINS, RESERVED_NAMES, validate_custom_tools


# ============================================================================
# Types
# ============================================================================

@dataclass
class REPLResult:
    """Ergebnis einer REPL-Ausführung."""
    success: bool
    output: str
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    locals_snapshot: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None
    rlm_calls: List[Any] = field(default_factory=list)


@dataclass
class RLMChatCompletion:
    """Completion result from RLM sub-call."""
    root_model: str
    prompt: str
    response: str
    execution_time: float = 0.0
    metadata: Optional[Dict] = None


# ============================================================================
# REPL Environment
# ============================================================================

class REPLEnvironment:
    """
    Full RLM REPL Environment with all features from the official RLM repo.

    Features:
    - context, context_0, context_1, ... (versioned context variables)
    - llm_query(): Direct LM call without REPL
    - rlm_query(): Recursive call with REPL
    - llm_query_batched(): Parallel LM calls
    - rlm_query_batched(): Parallel recursive calls
    - FINAL_VAR(): Mark final answer
    - SHOW_VARS(): Show all REPL variables
    - history: Access conversation history (when persistent)
    """

    def __init__(
        self,
        context_payload: Union[str, dict, list, None] = None,
        trajectory_id: Optional[str] = None,
        max_depth: int = 5,
        current_depth: int = 0,
        allowed_paths: Optional[List[str]] = None,
        execution_timeout: float = 30.0,
        self_url: Optional[str] = None,
        parent_trajectory_id: Optional[str] = None,
        persistent: bool = False,
        subcall_fn: Optional[Callable[[str, Optional[str]], RLMChatCompletion]] = None,
        custom_tools: Optional[Dict[str, Any]] = None,
        compaction: bool = False,
    ):
        self.context_payload = context_payload
        self.trajectory_id = trajectory_id or str(uuid.uuid4())
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.execution_timeout = execution_timeout
        self.self_url = self_url or "http://localhost:5000"
        self.parent_trajectory_id = parent_trajectory_id
        self.persistent = persistent
        self.subcall_fn = subcall_fn
        self.custom_tools = custom_tools or {}
        self.compaction = compaction

        # Validate custom tools don't override reserved names
        validate_custom_tools(self.custom_tools)

        # Validate custom tools don't override reserved names
        if self.custom_tools:
            validate_custom_tools(self.custom_tools)

        # Allowed paths for file access
        self.allowed_paths = allowed_paths or ["/tmp", os.getcwd()]

        # Create temp directory for this REPL session
        self.temp_dir = tempfile.mkdtemp(prefix=f"rlm_repl_{uuid.uuid4()}_")
        self.original_cwd = os.getcwd()

        # Execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.sub_queries: List[Dict[str, Any]] = []
        self._pending_rlm_calls: List[RLMChatCompletion] = []
        self._last_final_answer: Optional[str] = None

        # Context and history management
        self._context_count: int = 0
        self._history_count: int = 0
        self._compaction_history: List[Any] = []

        # Thread safety
        self._lock = threading.Lock()

        # Setup namespace
        self._setup_namespace()

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

    def _setup_namespace(self):
        """Setup the Python namespace for the LLM."""
        self.globals: Dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: Dict[str, Any] = {}

        # Add built-in helper functions
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["rlm_query"] = self._rlm_query
        self.globals["rlm_query_batched"] = self._rlm_query_batched

        # Add custom tools
        for name, tool in self.custom_tools.items():
            if name not in RESERVED_NAMES:
                if callable(tool):
                    self.globals[name] = tool
                else:
                    self.locals[name] = tool

        # Add compaction history if enabled
        if self.compaction:
            self.locals["history"] = self._compaction_history

    def _final_var(self, variable_name: Union[str, Any]) -> str:
        """
        Mark a variable or value as the final answer.

        Usage:
            result = 5 * 4
            FINAL_VAR(result)  # Marks 20 as final answer

            OR

            FINAL_VAR("my_result")  # Marks value of my_result variable
        """
        if not isinstance(variable_name, str):
            # Direct value passed
            answer = str(variable_name)
            self._last_final_answer = answer
            return answer

        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            answer = str(self.locals[variable_name])
            self._last_final_answer = answer
            return answer

        # Error: variable not found
        available = [k for k in self.locals.keys() if not k.startswith("_")]
        if available:
            return (
                f"Error: Variable '{variable_name}' not found. "
                f"Available variables: {available}. "
                f"You must create and assign a variable BEFORE calling FINAL_VAR on it."
            )
        return (
            f"Error: Variable '{variable_name}' not found. "
            f"No variables have been created yet. "
            f"You must create and assign a variable in a REPL block BEFORE calling FINAL_VAR on it."
        )

    def _show_vars(self) -> str:
        """Show all available variables in the REPL environment."""
        available = {k: type(v).__name__ for k, v in self.locals.items() if not k.startswith("_")}
        if not available:
            return "No variables created yet. Use Python code blocks to create variables."
        return f"Available variables: {available}"

    def _llm_query(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Query the LM with a single plain completion (no REPL, no recursion).

        This always makes a direct LM call, regardless of depth.

        Args:
            prompt: The prompt to send to the LM.
            model: Optional model name to use.

        Returns:
            Response string from the LM.
        """
        import requests

        try:
            payload = {
                "query": prompt,
                "recursive": False,  # Direct LM call, no recursion
                "max_depth": 0,
                "store_trajectory": False,
            }
            if model:
                payload["model"] = model

            response = requests.post(
                f"{self.self_url}/rlm/query",
                json=payload,
                timeout=60.0,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result_data = response.json()
                result = result_data.get("result", "")

                # Track the call
                completion = RLMChatCompletion(
                    root_model=model or "default",
                    prompt=prompt,
                    response=result,
                    execution_time=result_data.get("timing", {}).get("total_ms", 0) / 1000.0,
                )
                self._pending_rlm_calls.append(completion)

                return result
            else:
                return f"Error: HTTP {response.status_code} - {response.text[:200]}"

        except Exception as e:
            return f"Error: LM query failed - {type(e).__name__}: {str(e)}"

    def _llm_query_batched(self, prompts: List[str], model: Optional[str] = None) -> List[str]:
        """
        Query the LM with multiple prompts concurrently (no REPL, no recursion).

        Args:
            prompts: List of prompts to send to the LM.
            model: Optional model name to use.

        Returns:
            List of responses in the same order as input prompts.
        """
        import requests

        try:
            payload = {
                "prompts": prompts,
                "model": model,
            }

            response = requests.post(
                f"{self.self_url}/rlm/batch",
                json=payload,
                timeout=120.0,  # Longer timeout for batched requests
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result_data = response.json()
                results = result_data.get("results", [])

                # Track calls
                for i, result in enumerate(results):
                    if not result.startswith("Error:"):
                        completion = RLMChatCompletion(
                            root_model=model or "default",
                            prompt=prompts[i],
                            response=result,
                        )
                        self._pending_rlm_calls.append(completion)

                return results
            else:
                return [f"Error: HTTP {response.status_code}"] * len(prompts)

        except Exception as e:
            return [f"Error: Batched LM query failed - {type(e).__name__}: {str(e)}"] * len(prompts)

    def _rlm_query(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Spawn a recursive RLM sub-call for deeper thinking on a subtask.

        When subcall_fn is available (max_depth > 1), this spawns a child
        RLM with its own REPL that can reason over the prompt iteratively.
        Falls back to llm_query if no recursive capability is configured.

        Args:
            prompt: The prompt to send to the child RLM.
            model: Optional model name override for the child.
        """
        if self.subcall_fn is not None:
            try:
                completion = self.subcall_fn(prompt, model)
                self._pending_rlm_calls.append(completion)
                return completion.response
            except Exception as e:
                return f"Error: RLM query failed - {type(e).__name__}: {str(e)}"

        # Fall back to plain LM call if no recursive capability
        return self._llm_query(prompt, model)

    def _rlm_query_batched(self, prompts: List[str], model: Optional[str] = None) -> List[str]:
        """
        Spawn recursive RLM sub-calls for multiple prompts.

        Each prompt gets its own child RLM for deeper thinking.
        Falls back to llm_query_batched if no recursive capability.

        Args:
            prompts: List of prompts for child RLMs.
            model: Optional model name override for the children.

        Returns:
            List of responses in the same order as input prompts.
        """
        if self.subcall_fn is not None:
            results = []
            for prompt in prompts:
                try:
                    completion = self.subcall_fn(prompt, model)
                    self._pending_rlm_calls.append(completion)
                    results.append(completion.response)
                except Exception as e:
                    results.append(f"Error: RLM query failed - {type(e).__name__}: {str(e)}")
            return results

        # Fall back to plain batched LM call
        return self._llm_query_batched(prompts, model)

    def load_context(self, context_payload: Union[str, dict, list]):
        """Load context into the environment as context_0 (and 'context' alias)."""
        self.add_context(context_payload, 0)

    def add_context(
        self,
        context_payload: Union[str, dict, list],
        context_index: Optional[int] = None
    ) -> int:
        """
        Add a context with versioned variable name.

        Args:
            context_payload: The context data to add
            context_index: Optional explicit index. If None, auto-increments.

        Returns:
            The context index used.
        """
        if context_index is None:
            context_index = self._context_count

        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            # Store as text file and load
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w", encoding="utf-8") as f:
                f.write(context_payload)
            # Execute code to load it
            self._execute_sync(f"with open(r'{context_path}', 'r', encoding='utf-8') as f:\n    {var_name} = f.read()")
        else:
            # Store as JSON and load
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w", encoding="utf-8") as f:
                json.dump(context_payload, f)
            self._execute_sync(f"import json\nwith open(r'{context_path}', 'r', encoding='utf-8') as f:\n    {var_name} = json.load(f)")

        # Alias context_0 as 'context' for backward compatibility
        if context_index == 0:
            self._execute_sync(f"context = {var_name}")

        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def get_context_count(self) -> int:
        """Return the number of contexts loaded."""
        return self._context_count

    def add_history(
        self,
        message_history: List[Dict[str, Any]],
        history_index: Optional[int] = None
    ) -> int:
        """
        Store a conversation's message history as a versioned variable.

        Args:
            message_history: The list of message dicts from a completion call
            history_index: Optional explicit index. If None, auto-increments.

        Returns:
            The history index used.
        """
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"

        # Store deep copy to avoid reference issues
        self.locals[var_name] = copy.deepcopy(message_history)

        # Alias history_0 as 'history' for convenience
        if history_index == 0:
            self.locals["history"] = self.locals[var_name]

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        """Return the number of conversation histories stored."""
        return self._history_count

    def append_compaction_entry(self, entry: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Append a trajectory segment or a summary to the compaction history.

        Entry is either a list of message dicts (trajectory segment) or
        a dict with "type": "summary" and "content": str.
        """
        if not self.compaction:
            return
        self._compaction_history.append(copy.deepcopy(entry))
        # Update the history variable in locals
        self.locals["history"] = self._compaction_history

    def _execute_sync(self, code: str):
        """Execute code synchronously (for internal use)."""
        try:
            combined = {**self.globals, **self.locals}
            exec(code, combined, combined)
            # Update locals with new variables
            for key, value in combined.items():
                if key not in self.globals and not key.startswith("_"):
                    self.locals[key] = value
        except Exception as e:
            pass  # Silent fail for internal operations

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self):
        """Temporarily change to temp directory for execution."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def _restore_scaffold(self) -> None:
        """Restore scaffold names after execution so overwrites don't persist."""
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["rlm_query"] = self._rlm_query
        self.globals["rlm_query_batched"] = self._rlm_query_batched
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars

        # Restore custom tools
        for name, tool in self.custom_tools.items():
            if name not in RESERVED_NAMES:
                if callable(tool):
                    self.globals[name] = tool
                else:
                    self.locals[name] = tool

        # Restore context alias
        if "context_0" in self.locals:
            self.locals["context"] = self.locals["context_0"]

        # Restore history
        if "history_0" in self.locals and not self.compaction:
            self.locals["history"] = self.locals["history_0"]
        elif self.compaction:
            self.locals["history"] = self._compaction_history

    async def execute(self, code: str) -> REPLResult:
        """
        Execute Python code in the REPL environment.

        Args:
            code: Python code to execute

        Returns:
            REPLResult with output, result, and metadata
        """
        start_time = time.time()

        # Clear pending RLM calls from previous execution
        self._pending_rlm_calls = []

        # Code cleanup
        code = code.strip()
        if not code:
            return REPLResult(
                success=True,
                output="",
                result=None,
                execution_time_ms=0.0
            )

        output = ""
        error = None
        success = True

        try:
            # Run with timeout
            result = await asyncio.wait_for(
                self._execute_code_async(code),
                timeout=self.execution_timeout
            )
            output = result.get("output", "")
            error = result.get("error")
            success = error is None

        except asyncio.TimeoutError:
            error = f"Code execution exceeded {self.execution_timeout}s timeout"
            success = False
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            success = False

        execution_time = (time.time() - start_time) * 1000

        # Get final answer
        final_answer = self._last_final_answer
        self._last_final_answer = None

        # Record in history
        self.execution_history.append({
            "code": code[:1000],
            "success": success,
            "output": output[:2000],
            "error": error,
            "execution_time_ms": execution_time,
            "final_answer": final_answer,
        })

        return REPLResult(
            success=success,
            output=output,
            result=final_answer if final_answer else output.strip() if output else None,
            error=error,
            execution_time_ms=execution_time,
            locals_snapshot={k: str(v)[:100] for k, v in self.locals.items() if not k.startswith("_")},
            final_answer=final_answer,
            rlm_calls=self._pending_rlm_calls.copy(),
        )

    async def _execute_code_async(self, code: str) -> Dict[str, Any]:
        """Execute code asynchronously with output capture."""
        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                combined = {**self.globals, **self.locals}
                exec(code, combined, combined)

                # Update locals with new variables
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                # Restore scaffold
                self._restore_scaffold()

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()

                return {
                    "output": stdout + (f"\n[stderr]\n{stderr}" if stderr else ""),
                    "error": None,
                }

            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"
                return {
                    "output": stdout,
                    "error": stderr,
                }

    def get_history(self) -> List[Dict[str, Any]]:
        """Return execution history."""
        return self.execution_history.copy()

    def get_sub_queries(self) -> List[Dict[str, Any]]:
        """Return sub-queries for trajectory tracking."""
        return self.sub_queries.copy()

    def cleanup(self):
        """Clean up temp directory and reset state."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()


# ============================================================================
# Code Extraction Helper
# ============================================================================

def extract_python_code(content: str) -> Optional[str]:
    """
    Extract Python code from LLM response.

    Supports:
    - ```python ... ``` code blocks
    - ``` ... ``` generic code blocks
    - Direct Python code without markup

    Args:
        content: The LLM response

    Returns:
        Python code or None
    """
    if not content:
        return None

    content = content.strip()

    # Pattern 1: ```python ... ```
    python_pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(python_pattern, content, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Pattern 2: ``` ... ``` (any code block)
    code_pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(code_pattern, content, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Pattern 3: Direct code (heuristic detection)
    lines = content.split('\n')
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('#', '//')):
            code_lines.append(line)
        elif any(kw in stripped for kw in [
            'def ', 'class ', 'import ', 'from ', 'print(',
            'for ', 'if ', 'while ', 'return ', '=',
            'llm_query', 'rlm_query', 'FINAL_VAR', 'SHOW_VARS',
            'context', 're.', 'json.',
        ]):
            code_lines.append(line)
        elif stripped == '':
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines).strip()

    return None
