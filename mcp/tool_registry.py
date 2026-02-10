from typing import Callable, Dict, Any

TOOL_REGISTRY: Dict[str, Callable[[dict], Any]] = {}


def register_tool(name: str):
    def wrapper(func: Callable[[dict], Any]):
        TOOL_REGISTRY[name] = func
        return func
    return wrapper
