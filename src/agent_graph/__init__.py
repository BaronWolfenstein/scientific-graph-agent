
"""Agent Graph for scientific paper exploration."""
from .state import InputState, InternalState, PrivateState, OutputState

# Defer graph imports to avoid requiring langgraph for modules that don't need it
def __getattr__(name):
    if name == "create_graph":
        from .graph import create_graph
        return create_graph
    elif name == "create_streaming_graph":
        from .graph import create_streaming_graph
        return create_streaming_graph
    elif name == "create_graph_with_approval":
        from .graph import create_graph_with_approval
        return create_graph_with_approval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "create_graph",
    "create_streaming_graph",
    "create_graph_with_approval",
    "InputState",
    "InternalState",
    "PrivateState",
    "OutputState"
]