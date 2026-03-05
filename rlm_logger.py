"""
RLM Logger - Trajectory Logging and Visualization Support
===========================================================

Implements logging for RLM trajectories with JSONL export for the visualizer.
Based on the official RLM implementation.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class RLMIteration:
    """Single iteration of RLM execution."""
    step: int
    prompt: str
    response: str
    code_blocks: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RLMMetadata:
    """Metadata for an RLM run."""
    root_model: str
    max_depth: int
    max_iterations: int
    backend: str
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)
    environment_type: str = "local"
    environment_kwargs: Dict[str, Any] = field(default_factory=dict)
    other_backends: Optional[List[str]] = None
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RLMChatCompletion:
    """Result of an RLM completion."""
    root_model: str
    prompt: Union[str, Dict[str, Any]]
    response: str
    usage_summary: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None


class RLMLogger:
    """
    Logger for RLM trajectories.

    Supports:
    - In-memory logging (trajectory available on completion.metadata)
    - Disk persistence (JSONL for visualizer)
    - Trajectory reconstruction
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize logger.

        Args:
            log_dir: Optional directory to save JSONL files. If None, only in-memory logging.
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.iterations: List[RLMIteration] = []
        self.metadata: Optional[RLMMetadata] = None
        self._current_trajectory_file: Optional[Path] = None

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_metadata(self, metadata: RLMMetadata):
        """Log metadata for this RLM run."""
        self.metadata = metadata

        if self.log_dir:
            # Create trajectory file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{metadata.trajectory_id[:8]}_{timestamp}.jsonl"
            self._current_trajectory_file = self.log_dir / filename

            # Write metadata as first line
            self._append_to_file({
                "type": "metadata",
                "data": asdict(metadata),
            })

    def log(self, iteration: Union[RLMIteration, Dict[str, Any]]):
        """Log an iteration."""
        if isinstance(iteration, dict):
            iteration = RLMIteration(**iteration)

        self.iterations.append(iteration)

        if self.log_dir and self._current_trajectory_file:
            self._append_to_file({
                "type": "iteration",
                "data": asdict(iteration),
            })

    def log_subcall(self, parent_depth: int, prompt: str, response: str, execution_time: float):
        """Log a sub-call (recursive RLM call)."""
        if self.log_dir and self._current_trajectory_file:
            self._append_to_file({
                "type": "subcall",
                "data": {
                    "parent_depth": parent_depth,
                    "prompt": prompt[:500] if len(prompt) > 500 else prompt,
                    "response": response[:500] if len(response) > 500 else response,
                    "execution_time": execution_time,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            })

    def log_code_execution(self, code: str, output: str, success: bool, error: Optional[str] = None):
        """Log a code execution block."""
        if self.log_dir and self._current_trajectory_file:
            self._append_to_file({
                "type": "code_execution",
                "data": {
                    "code": code[:1000],
                    "output": output[:2000],
                    "success": success,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            })

    def _append_to_file(self, entry: Dict[str, Any]):
        """Append entry to JSONL file."""
        if self._current_trajectory_file:
            with open(self._current_trajectory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    def get_trajectory(self) -> Dict[str, Any]:
        """Get full trajectory as dictionary."""
        return {
            "metadata": asdict(self.metadata) if self.metadata else None,
            "iterations": [asdict(i) for i in self.iterations],
            "iteration_count": len(self.iterations),
        }

    def get_iterations(self) -> List[RLMIteration]:
        """Get all logged iterations."""
        return self.iterations.copy()

    def clear_iterations(self):
        """Clear iteration history."""
        self.iterations = []

    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """
        Save trajectory to a JSON file.

        Args:
            filepath: Optional file path. If None, uses log_dir.

        Returns:
            Path to saved file.
        """
        if filepath is None:
            if self.log_dir is None:
                raise ValueError("No log_dir set and no filepath provided")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            traj_id = self.metadata.trajectory_id[:8] if self.metadata else "unknown"
            filepath = self.log_dir / f"trajectory_{traj_id}_{timestamp}.json"

        trajectory = self.get_trajectory()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, indent=2, default=str)

        return str(filepath)

    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Any]:
        """Load trajectory from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
        """Load entries from JSONL file."""
        entries = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries


class TrajectoryVisualizer:
    """
    Simple visualizer for RLM trajectories.

    Generates HTML for viewing trajectories.
    """

    def __init__(self, trajectory: Dict[str, Any]):
        self.trajectory = trajectory

    def to_html(self) -> str:
        """Generate HTML representation of trajectory."""
        metadata = self.trajectory.get("metadata", {})
        iterations = self.trajectory.get("iterations", [])

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RLM Trajectory {metadata.get('trajectory_id', 'Unknown')[:8]}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .iteration {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .step-number {{
            font-size: 14px;
            color: #666;
            font-weight: 600;
        }}
        .prompt {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
        }}
        .response {{
            background: #e3f2fd;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
        }}
        .code-block {{
            background: #263238;
            color: #aed581;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
        }}
        .metadata {{
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        h2 {{
            margin: 0 0 15px 0;
            color: #555;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RLM Trajectory</h1>
        <div class="metadata">
            <strong>Model:</strong> {metadata.get('root_model', 'Unknown')}<br>
            <strong>Max Depth:</strong> {metadata.get('max_depth', 'N/A')}<br>
            <strong>Backend:</strong> {metadata.get('backend', 'N/A')}<br>
            <strong>Created:</strong> {metadata.get('created_at', 'N/A')}
        </div>
    </div>
"""

        for i, iteration in enumerate(iterations):
            html += f"""
    <div class="iteration">
        <div class="step-number">Step {iteration.get('step', i + 1)}</div>
        <div class="prompt"><strong>Prompt:</strong><br>{self._escape_html(iteration.get('prompt', ''))}</div>
        <div class="response"><strong>Response:</strong><br>{self._escape_html(iteration.get('response', ''))}</div>
"""
            # Add code blocks if present
            code_blocks = iteration.get('code_blocks', [])
            for j, block in enumerate(code_blocks):
                html += f"""
        <div class="code-block">
            <strong>Code Block {j + 1}:</strong><br>
            {self._escape_html(block.get('code', ''))}
        </div>
"""

            html += f"""
        <div class="metadata">
            Execution time: {iteration.get('execution_time_ms', 0):.0f}ms
        </div>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))

    def save_html(self, filepath: str):
        """Save HTML visualization to file."""
        html = self.to_html()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
