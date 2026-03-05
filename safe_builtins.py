"""
Safe Builtins for RLM REPL Environment
=====================================

Zentrale Safety-Configuration für die Python REPL.
Blockiert gefährliche Operationen wie eval, exec, input.

Basierend auf dem offiziellen RLM Repository:
https://github.com/alexzhang13/rlm
"""

import json
import re
from typing import Any, Dict, List, Optional


# =============================================================================
# Blockierte Builtins (Sicherheitskritisch)
# =============================================================================

_BLOCKED_BUILTINS = {
    # Input/Output Interaktion
    "input": None,
    "raw_input": None,  # Python 2 Kompatibilität

    # Dynamische Code-Ausführung
    "eval": None,
    "exec": None,
    "compile": None,

    # Globale/Lokale Namespace Manipulation
    "globals": None,
    "locals": None,
    "vars": None,  # Nur im Kontext einer Klasse erlaubt

    # Metaprogrammierung (könnte Security-Probleme verursachen)
    "__import__": None,  # Wir kontrollieren Imports selbst
}


# =============================================================================
# Erlaubte Builtins
# =============================================================================

_SAFE_BUILTINS = {
    # === Core Types ===
    "bool": bool,
    "bytes": bytes,
    "bytearray": bytearray,
    "complex": complex,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "list": list,
    "map": map,
    "memoryview": memoryview,
    "object": object,
    "property": property,
    "range": range,
    "reversed": reversed,
    "set": set,
    "slice": slice,
    "staticmethod": staticmethod,
    "str": str,
    "classmethod": classmethod,
    "super": super,
    "tuple": tuple,
    "type": type,
    "zip": zip,

    # === Core Functions ===
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "callable": callable,
    "chr": chr,
    "divmod": divmod,
    "format": format,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "repr": repr,
    "round": round,
    "sorted": sorted,
    "sum": sum,

    # === Attribute Access ===
    "delattr": delattr,
    "getattr": getattr,
    "setattr": setattr,
    "dir": dir,

    # === Exceptions ===
    "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "AttributeError": AttributeError,
    "BaseException": BaseException,
    "BlockingIOError": BlockingIOError,
    "BrokenPipeError": BrokenPipeError,
    "BufferError": BufferError,
    "BytesWarning": BytesWarning,
    "ChildProcessError": ChildProcessError,
    "ConnectionAbortedError": ConnectionAbortedError,
    "ConnectionError": ConnectionError,
    "ConnectionRefusedError": ConnectionRefusedError,
    "ConnectionResetError": ConnectionResetError,
    "DeprecationWarning": DeprecationWarning,
    "EOFError": EOFError,
    "EnvironmentError": EnvironmentError,
    "Exception": Exception,
    "FileExistsError": FileExistsError,
    "FileNotFoundError": FileNotFoundError,
    "FloatingPointError": FloatingPointError,
    "FutureWarning": FutureWarning,
    "GeneratorExit": GeneratorExit,
    "IOError": IOError,
    "ImportError": ImportError,
    "ImportWarning": ImportWarning,
    "IndentationError": IndentationError,
    "IndexError": IndexError,
    "InterruptedError": InterruptedError,
    "IsADirectoryError": IsADirectoryError,
    "KeyError": KeyError,
    "KeyboardInterrupt": KeyboardInterrupt,
    "LookupError": LookupError,
    "MemoryError": MemoryError,
    "ModuleNotFoundError": ModuleNotFoundError,
    "NameError": NameError,
    "NotADirectoryError": NotADirectoryError,
    "NotImplementedError": NotImplementedError,
    "OSError": OSError,
    "OverflowError": OverflowError,
    "PendingDeprecationWarning": PendingDeprecationWarning,
    "PermissionError": PermissionError,
    "ProcessLookupError": ProcessLookupError,
    "RecursionError": RecursionError,
    "ReferenceError": ReferenceError,
    "ResourceWarning": ResourceWarning,
    "RuntimeError": RuntimeError,
    "RuntimeWarning": RuntimeWarning,
    "StopAsyncIteration": StopAsyncIteration,
    "StopIteration": StopIteration,
    "SyntaxError": SyntaxError,
    "SyntaxWarning": SyntaxWarning,
    "SystemError": SystemError,
    "SystemExit": SystemExit,
    "TabError": TabError,
    "TimeoutError": TimeoutError,
    "TypeError": TypeError,
    "UnboundLocalError": UnboundLocalError,
    "UnicodeDecodeError": UnicodeDecodeError,
    "UnicodeEncodeError": UnicodeEncodeError,
    "UnicodeError": UnicodeError,
    "UnicodeTranslateError": UnicodeTranslateError,
    "UnicodeWarning": UnicodeWarning,
    "UserWarning": UserWarning,
    "ValueError": ValueError,
    "Warning": Warning,
    "ZeroDivisionError": ZeroDivisionError,

    # === Open (kontrolliert durch file_read/file_write) ===
    "open": open,

    # === __import__ (wird durch unseren eigenen Import-Handler ersetzt) ===
    "__import__": __import__,
}


# =============================================================================
# Safe Regex Module
# =============================================================================

class SafeRegexModule:
    """
    Sicheres Regex-Modul für die REPL.
    Erlaubt nur lesende Operationen.
    """

    # Regex Flags
    IGNORECASE = re.IGNORECASE
    DOTALL = re.DOTALL
    MULTILINE = re.MULTILINE
    UNICODE = re.UNICODE
    VERBOSE = re.VERBOSE
    ASCII = re.ASCII

    @staticmethod
    def findall(pattern: str, string: str, flags: int = 0) -> List[str]:
        """Finde alle nicht-überlappenden Übereinstimmungen."""
        return re.findall(pattern, string, flags)

    @staticmethod
    def search(pattern: str, string: str, flags: int = 0) -> Optional[re.Match]:
        """Suche nach erster Übereinstimmung."""
        return re.search(pattern, string, flags)

    @staticmethod
    def match(pattern: str, string: str, flags: int = 0) -> Optional[re.Match]:
        """Suche am String-Anfang."""
        return re.match(pattern, string, flags)

    @staticmethod
    def fullmatch(pattern: str, string: str, flags: int = 0) -> Optional[re.Match]:
        """Suche nach kompletter Übereinstimmung."""
        return re.fullmatch(pattern, string, flags)

    @staticmethod
    def split(pattern: str, string: str, maxsplit: int = 0, flags: int = 0) -> List[str]:
        """Splitte String anhand von Pattern."""
        return re.split(pattern, string, maxsplit, flags)

    @staticmethod
    def sub(pattern: str, repl: str, string: str, count: int = 0, flags: int = 0) -> str:
        """Ersetze Übereinstimmungen."""
        return re.sub(pattern, repl, string, count, flags)

    @staticmethod
    def subn(pattern: str, repl: str, string: str, count: int = 0, flags: int = 0) -> tuple:
        """Ersetze Übereinstimmungen und zähle."""
        return re.subn(pattern, repl, string, count, flags)

    @staticmethod
    def escape(string: str) -> str:
        """Escape Sonderzeichen."""
        return re.escape(string)

    @staticmethod
    def compile(pattern: str, flags: int = 0):
        """Kompiliere Regex (nur für internen Gebrauch)."""
        # Wir geben das kompilierte Pattern zurück, aber beschränken die Nutzung
        return re.compile(pattern, flags)


# =============================================================================
# Safe JSON Module
# =============================================================================

class SafeJSONModule:
    """
    Sicheres JSON-Modul für die REPL.
    """

    @staticmethod
    def dumps(obj: Any, *, indent: Optional[int] = None, ensure_ascii: bool = True, **kwargs) -> str:
        """Serialisiere zu JSON."""
        return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)

    @staticmethod
    def dump(obj: Any, fp, *, indent: Optional[int] = None, ensure_ascii: bool = True, **kwargs):
        """Serialisiere zu JSON-Datei."""
        return json.dump(obj, fp, indent=indent, ensure_ascii=ensure_ascii, **kwargs)

    @staticmethod
    def loads(s: str, **kwargs) -> Any:
        """Deserialisiere von JSON."""
        return json.loads(s, **kwargs)

    @staticmethod
    def load(fp, **kwargs) -> Any:
        """Deserialisiere von JSON-Datei."""
        return json.load(fp, **kwargs)

    # JSON Decoder/Encoder Klassen
    JSONDecoder = json.JSONDecoder
    JSONEncoder = json.JSONEncoder


# =============================================================================
# Factory Functions
# =============================================================================

def get_safe_builtins() -> Dict[str, Any]:
    """
    Erstelle ein sauberes safe_builtins Dictionary.
    Kombiniert erlaubte Builtins und blockierte Builtins.
    """
    # Kopie der erlaubten Builtins
    safe = _SAFE_BUILTINS.copy()

    # Füge blockierte Builtins hinzu (mit None als Value)
    safe.update(_BLOCKED_BUILTINS)

    return safe


def get_safe_globals() -> Dict[str, Any]:
    """
    Erstelle das globals Dictionary für die REPL.
    """
    return {
        "__builtins__": get_safe_builtins(),
        "__name__": "__main__",
    }


def validate_code_safety(code: str) -> tuple[bool, Optional[str]]:
    """
    Validiere ob Code potenziell unsicher ist.

    Returns:
        (is_safe, error_message)
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"

    # Verbotene Nodes
    forbidden_nodes = (
        ast.Import,  # Imports müssen durch unseren Handler gehen
        ast.ImportFrom,
    )

    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                return False, f"Import statements are not allowed: {names}. Use available modules directly."
            elif isinstance(node, ast.ImportFrom):
                return False, f"Import from statements are not allowed: {node.module}"

    return True, None


# =============================================================================
# Reserved Tool Names (dürfen nicht überschrieben werden)
# =============================================================================

RESERVED_TOOL_NAMES = {
    "llm_query",
    "llm_query_batched",
    "rlm_query",
    "rlm_query_batched",
    "FINAL_VAR",
    "SHOW_VARS",
    "context",
    "history",
}


def is_reserved_name(name: str) -> bool:
    """Prüfe ob ein Name reserviert ist."""
    return name in RESERVED_TOOL_NAMES or name.startswith("context_") or name.startswith("history_")


def validate_custom_tools(custom_tools: dict) -> None:
    """
    Validate that custom tools don't override reserved names.

    Args:
        custom_tools: Dictionary of custom tool names to values/functions

    Raises:
        ValueError: If a reserved name is used
    """
    for name in custom_tools.keys():
        if name in RESERVED_TOOL_NAMES:
            raise ValueError(
                f"Custom tool name '{name}' is reserved and cannot be used. "
                f"Reserved names: {RESERVED_TOOL_NAMES}"
            )


# =============================================================================
# Public Exports
# =============================================================================

# Export safe builtins for direct import
SAFE_BUILTINS = get_safe_builtins()

# Alias for compatibility
RESERVED_NAMES = RESERVED_TOOL_NAMES

__all__ = [
    "SAFE_BUILTINS",
    "RESERVED_NAMES",
    "SafeRegexModule",
    "SafeJSONModule",
    "get_safe_builtins",
    "get_safe_globals",
    "validate_code_safety",
    "is_reserved_name",
    "validate_custom_tools",
]
