from abc import ABC, abstractmethod
from typing import Dict, Any, Type
from pydantic import BaseModel

class GZToolInputSchema(BaseModel):
    """Base class for tool input schemas."""
    pass

class GZTool(ABC):
    """Abstract base class for all tools."""
    toolName: str
    description: str
    inputSchema: Type[GZToolInputSchema]

    def __init__(self, toolName: str, description: str, inputSchema: Type[GZToolInputSchema]):
        self.toolName = toolName
        self.description = description
        self.inputSchema = inputSchema

    @abstractmethod
    async def executeTool(self, validatedInput: GZToolInputSchema, **kwargs) -> Any:
        """Executes the tool with validated input."""
        pass

    def getJsonSchema(self) -> Dict[str, Any]:
        """Returns the JSON schema for the tool, used for LLM prompting."""
        return {
            "name": self.toolName,
            "description": self.description,
            "input_schema": self.inputSchema.model_json_schema()
        }