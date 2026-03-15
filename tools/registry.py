"""
ToolRegistry – central hub that wires all tool functions together and
exposes them to the agent as a name→callable map plus JSON schemas.
"""

from __future__ import annotations

from tools.radar_processing import load_radar_sequence, extract_radar_features
from tools.joint_analysis    import analyze_joint_motion
from tools.data_retrieval    import get_motion_text, search_motions
from tools.comparison        import compare_motions
from tools.visualization     import visualize_motion


class ToolRegistry:
    """Provides {name: callable} and JSON schemas for all agent tools."""

    def __init__(self) -> None:
        self._map: dict[str, callable] = {
            "load_radar_sequence":    load_radar_sequence,
            "extract_radar_features": extract_radar_features,
            "get_motion_text":        get_motion_text,
            "search_motions":         search_motions,
            "compare_motions":        compare_motions,
            "analyze_joint_motion":   analyze_joint_motion,
            "visualize_motion":       visualize_motion,
        }

    def get_tool_map(self) -> dict[str, callable]:
        """Return mapping of tool name → callable."""
        return dict(self._map)

    def get_tool_schemas(self) -> list:
        """Return the list of JSON schemas for Qwen 3 function calling."""
        from agent.tool_schemas import TOOL_SCHEMAS
        return TOOL_SCHEMAS

    def call(self, name: str, arguments: dict):
        """Dispatch a tool call by name.

        Returns the tool's result dict, or an error dict on failure.
        """
        fn = self._map.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name!r}"}
        try:
            return fn(**arguments)
        except Exception as exc:
            return {"error": f"Tool {name!r} raised {type(exc).__name__}: {exc}"}
