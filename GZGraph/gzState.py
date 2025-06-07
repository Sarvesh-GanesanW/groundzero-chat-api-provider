from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
import json


class GZMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    speaker: str 
    content: Any
    contentType: str = Field(default="text")
    toolCalls: Optional[List[Dict[str, Any]]] = None 
    toolCallId: Optional[str] = None 
    toolName: Optional[str] = None 
    isError: bool = False

class GZToolCallRequest(BaseModel):
    toolCallId: str = Field(default_factory=lambda: f"toolcall_{str(uuid.uuid4())}")
    toolName: str
    toolInput: Dict[str, Any]

class GZToolCallResult(BaseModel):
    toolCallId: str
    toolName: str
    toolOutput: Any
    isError: bool = False
    errorMessage: Optional[str] = None

class GZAgentState(BaseModel):
    userInput: str
    conversationId: str
    userId: str
    modelId: str
    messages: List[GZMessage] = Field(default_factory=list)
    ragContext: Optional[str] = None
    
    requestedToolCalls: List[GZToolCallRequest] = Field(default_factory=list)
    executedToolResults: List[GZToolCallResult] = Field(default_factory=list)
    
    finalOutput: Optional[str] = None
    internalScratchpad: Dict[str, Any] = Field(default_factory=dict)
    errorOccurred: bool = False
    errorMessage: Optional[str] = None
    currentStepOutput: Any = None 

    def addMessage(self, speaker: str, content: Any, contentType: str = "text", toolCalls: Optional[List[Dict[str, Any]]] = None, isError: bool = False, toolCallId: Optional[str] = None, toolName: Optional[str] = None):
        self.messages.append(GZMessage(
            speaker=speaker,
            content=content,
            contentType=contentType,
            toolCalls=toolCalls,
            isError=isError,
            toolCallId=toolCallId,
            toolName=toolName
        ))

    def addToolCallRequest(self, toolName: str, toolInput: Dict[str, Any]) -> str:
        request = GZToolCallRequest(toolName=toolName, toolInput=toolInput)
        self.requestedToolCalls.append(request)
        return request.toolCallId

    def addToolCallResult(self, toolCallId: str, toolName: str, toolOutput: Any, isError: bool = False, errorMessage: Optional[str] = None):
        self.executedToolResults.append(GZToolCallResult(
            toolCallId=toolCallId,
            toolName=toolName,
            toolOutput=toolOutput,
            isError=isError,
            errorMessage=errorMessage
        ))
        self.requestedToolCalls = [tc for tc in self.requestedToolCalls if tc.toolCallId != toolCallId]


    def getLastUserMessageContent(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.speaker == "user" and msg.contentType == "text":
                return msg.content
        return self.userInput 

    def getFormattedHistoryForPrompt(self, maxMessages: int = 10) -> str:
        historyLines = []
        relevantMessages = self.messages[-maxMessages:]
        for msg in relevantMessages:
            prefix = ""
            if msg.speaker == "user":
                prefix = "Human"
            elif msg.speaker == "ai":
                prefix = "Assistant"
                if msg.toolCalls:
                    historyLines.append(f"{prefix}: (Thinking... Decided to use tools: {json.dumps(msg.toolCalls)})")
                    continue 
            elif msg.speaker == "tool":
                prefix = f"Tool Response ({msg.toolName} ID: {msg.toolCallId})"
            elif msg.speaker == "system":
                prefix = "System"
            
            if prefix:
                 historyLines.append(f"{prefix}: {msg.content if isinstance(msg.content, str) else json.dumps(msg.content)}")
        return "\n".join(historyLines)