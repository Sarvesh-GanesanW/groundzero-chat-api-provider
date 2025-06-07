from typing import Dict, Any, Optional, List
from GZGraph.tools import AWSLambdaTool, CurrentDateTimeTool, AsyncDDGSearchTool, AsyncWebScraperTool

class GZToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def registerTool(self, tool: Any):
        if tool.toolName in self.tools:
            raise ValueError(f"Tool with name '{tool.toolName}' already registered.")
        self.tools[tool.toolName] = tool

    def getTool(self, toolName: str) -> Optional[Any]:
        return self.tools.get(toolName)

    def getAvailableToolsSchemas(self) -> List[Dict[str, Any]]:
        return [tool.getJsonSchema() for tool in self.tools.values()]

defaultToolRegistry = GZToolRegistry()
defaultToolRegistry.registerTool(AsyncDDGSearchTool())
defaultToolRegistry.registerTool(AsyncWebScraperTool())
defaultToolRegistry.registerTool(CurrentDateTimeTool())
defaultToolRegistry.registerTool(AWSLambdaTool())
