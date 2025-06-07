import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, AsyncGenerator, Any, List
from pydantic import ValidationError
import uuid
from GZGraph.gzState import GZAgentState, GZToolCallRequest
from GZGraph.gzToolRegistry import defaultToolRegistry
from utils.bedrockUtils import getEmbeddingsBedrock, interactWithBedrock
from utils.databaseUtils import findSimilarChunks
from constants import Constants
import re

class GZNode(ABC):
    nodeName: str

    def __init__(self, nodeName: str):
        self.nodeName = nodeName

    @abstractmethod
    async def execute(self, state: GZAgentState, config: Optional[Dict] = None) -> AsyncGenerator[Dict[str, Any], GZAgentState]:
        pass

    async def streamExecute(self, state: GZAgentState, config: Optional[Dict] = None) -> AsyncGenerator[Dict[str, Any], GZAgentState]:
        updatedState = await self.execute(state, config)
        yield {"type": "node_complete", "nodeName": self.nodeName, "data": updatedState.currentStepOutput}
        state.currentStepOutput = None
        return

class QueryPlanningNode(GZNode):
    async def execute(self, state: GZAgentState, config: Optional[Dict] = None) -> GZAgentState:
        userQuery = state.getLastUserMessageContent()

        planningPrompt = f"""
        Based on the user query: "{userQuery}",
        determine the best search strategy:

        1. What are the key concepts that need to be retrieved?
        2. What would be the most effective search queries for each concept?
        3. Should we retrieve documents in a specific order?
        4. Is there ambiguity that requires clarification before searching?

        Output your reasoning and a structured plan for retrieval.
        """

        response_stream = interactWithBedrock(planningPrompt, state.modelId)
        if response_stream is None:
            state.errorOccurred = True
            state.errorMessage = "Failed to get response from planning LLM"
            return state

        response = ""
        try:
            for event in response_stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_data = json.loads(chunk.get('bytes').decode())
                    if chunk_data.get('type') == 'content_block_delta':
                        delta = chunk_data.get('delta', {})
                        text = delta.get('text', '')
                        if text:
                            response += text
        except Exception as e:
            state.errorOccurred = True
            state.errorMessage = f"Error processing response stream: {str(e)}"
            return state

        state.internalScratchpad["retrievalPlan"] = response
        state.addMessage(speaker="system", content=f"Developed retrieval plan: {response[:100]}...")

        structuredQueries = []
        lines = response.split('\n')
        for line in lines:
            if re.search(r'query|search|ask|find', line.lower()):
                query = re.sub(r'^[^"\']*["\']|["\']$', '', line).strip()
                if query and len(query) > 5:
                    structuredQueries.append(query)

        structuredQueries = structuredQueries[:3]
        state.internalScratchpad["structuredQueries"] = structuredQueries

        return state

class InitialProcessingNode(GZNode):
    async def execute(self, state: GZAgentState, config: Optional[Dict] = None) -> GZAgentState:
        state.addMessage(speaker="user", content=state.userInput)
        state.currentStepOutput = {"message": "User input processed"}
        return state

class RAGNode(GZNode):
    async def execute(self, siteName, state: GZAgentState, config: Optional[Dict] = None) -> GZAgentState:
        if config and config.get("disableRag") is True:
            state.currentStepOutput = {"message": "RAG processing was disabled for this step."}
            return state

        lastUserMessage = state.getLastUserMessageContent()
        if not lastUserMessage:
            state.currentStepOutput = {"message": "No user message for RAG."}
            return state

        try:
            queryEmbedding = getEmbeddingsBedrock([lastUserMessage])[0]
            if queryEmbedding:
                rag_top_k = 3
                if config and config.get("ragTopK"):
                    rag_top_k = config.get("ragTopK")

                similarChunks = findSimilarChunks(queryEmbedding, limit=rag_top_k, siteName=siteName)
                if similarChunks:
                    state.ragContext = "\n\nRelevant information found:\n" + "\n---\n".join(similarChunks)
                    state.addMessage(speaker="system", content=f"Retrieved RAG context based on: '{lastUserMessage[:50]}...'")
                    state.currentStepOutput = {"ragContextLength": len(state.ragContext), "numChunks": len(similarChunks)}
                else:
                    state.currentStepOutput = {"message": "No relevant chunks found by RAG."}
            else:
                state.currentStepOutput = {"message": "Failed to generate embedding for RAG."}
        except Exception as e:
            state.errorOccurred = True
            state.errorMessage = f"RAGNode error: {str(e)}"
            state.addMessage(speaker="system", content=f"Error during RAG processing: {str(e)}", isError=True)
            state.currentStepOutput = {"error": state.errorMessage}
        return state

class EnhancedRAGNode(RAGNode):
    async def execute(self, siteName, state: GZAgentState, config: Optional[Dict] = None) -> GZAgentState:
        if config and config.get("disableRag") is True:
            state.currentStepOutput = {"message": "RAG processing was disabled for this step."}
            return state

        structuredQueries = state.internalScratchpad.get("structuredQueries", [])

        if not structuredQueries:
            lastUserMessage = state.getLastUserMessageContent()
            structuredQueries = [lastUserMessage] if lastUserMessage else []

        if not structuredQueries:
            state.currentStepOutput = {"message": "No queries for RAG."}
            return state

        allRetrievedChunks = []

        try:
            for query in structuredQueries:
                queryEmbedding = getEmbeddingsBedrock([query])[0]
                if queryEmbedding:
                    rag_top_k = 3
                    if config and config.get("ragTopK"):
                        rag_top_k = config.get("ragTopK")

                    similarChunks = findSimilarChunks(queryEmbedding, siteName, limit=rag_top_k)
                    if similarChunks:
                        taggedChunks = [f"Query: '{query}'\nResult: {chunk}" for chunk in similarChunks]
                        allRetrievedChunks.extend(taggedChunks)

            uniqueChunks = list(set(allRetrievedChunks))

            if uniqueChunks:
                state.ragContext = "\n\nRelevant information found:\n" + "\n---\n".join(uniqueChunks)
                state.addMessage(
                    speaker="system",
                    content=f"Retrieved {len(uniqueChunks)} relevant chunks using {len(structuredQueries)} queries."
                )
                state.currentStepOutput = {
                    "ragContextLength": len(state.ragContext),
                    "numChunks": len(uniqueChunks),
                    "numQueries": len(structuredQueries)
                }
            else:
                state.currentStepOutput = {"message": "No relevant chunks found by RAG."}
        except Exception as e:
            state.errorOccurred = True
            state.errorMessage = f"EnhancedRAGNode error: {str(e)}"
            state.addMessage(speaker="system", content=f"Error during RAG processing: {str(e)}", isError=True)
            state.currentStepOutput = {"error": state.errorMessage}

        return state

class ContextAnalysisNode(GZNode):
    async def execute(self, state: GZAgentState, config: Optional[Dict] = None) -> GZAgentState:
        userQuery = state.getLastUserMessageContent()
        retrievedContext = state.ragContext

        if not retrievedContext:
            state.addMessage(speaker="system", content="No context to analyze.")
            return state

        analysisPrompt = f"""
        User Query: "{userQuery}"

        Retrieved Context:
        {retrievedContext[:2000]}... [truncated]

        Analyze the retrieved information:
        1. Is the information sufficient to answer the query? If not, what's missing?
        2. Is there contradictory information that needs resolution?
        3. What additional searches or refinements would improve the answer?
        4. Rate the relevance of retrieved information on a scale of 1-10.

        Provide your analysis and recommendations.
        """

        response_stream = interactWithBedrock(analysisPrompt, state.modelId)
        if response_stream is None:
            state.errorOccurred = True
            state.errorMessage = "Failed to get response from analysis LLM"
            return state

        response = ""
        try:
            for event in response_stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_data = json.loads(chunk.get('bytes').decode())
                    if chunk_data.get('type') == 'content_block_delta':
                        delta = chunk_data.get('delta', {})
                        text = delta.get('text', '')
                        if text:
                            response += text
        except Exception as e:
            state.errorOccurred = True
            state.errorMessage = f"Error processing response stream: {str(e)}"
            return state

        state.internalScratchpad["contextAnalysis"] = response
        state.addMessage(speaker="system", content=f"Context analysis completed: {response[:100]}...")

        needsMoreInfo = "insufficient" in response.lower() or "missing" in response.lower()
        state.internalScratchpad["needsMoreInfo"] = needsMoreInfo

        return state

class InformationEnrichmentNode(GZNode):
    async def execute(self, state: GZAgentState, config: Optional[Dict] = None) -> GZAgentState:
        needsMoreInfo = state.internalScratchpad.get("needsMoreInfo", False)

        if not needsMoreInfo:
            return state

        analysis = state.internalScratchpad.get("contextAnalysis", "")
        userQuery = state.getLastUserMessageContent()

        enrichmentPrompt = f"""
        Based on the user query: "{userQuery}"

        And our analysis of available information:
        {analysis[:500]}...

        Generate up to 3 specific web search queries that would help fill the information gaps.
        Format each query on a new line starting with "QUERY: "
        """

        response_stream = interactWithBedrock(enrichmentPrompt, state.modelId)
        if response_stream is None:
            state.errorOccurred = True
            state.errorMessage = "Failed to get response from enrichment LLM"
            return state

        response = ""
        try:
            for event in response_stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_data = json.loads(chunk.get('bytes').decode())
                    if chunk_data.get('type') == 'content_block_delta':
                        delta = chunk_data.get('delta', {})
                        text = delta.get('text', '')
                        if text:
                            response += text
        except Exception as e:
            state.errorOccurred = True
            state.errorMessage = f"Error processing response stream: {str(e)}"
            return state

        searchQueries = []
        for line in response.split('\n'):
            if line.startswith("QUERY:"):
                query = line.replace("QUERY:", "").strip()
                if query:
                    searchQueries.append(query)

        state.internalScratchpad["webSearchQueries"] = searchQueries

        additionalContext = []
        for query in searchQueries:
            searchToolCallId = state.addToolCallRequest("webSearch", {"query": query, "numResults": 5})
            searchResults = [{"title": "Example Result", "snippet": "This is where search results would appear."}]

            state.addToolCallResult(searchToolCallId, "webSearch", searchResults)
            additionalContext.append(f"Web Search for '{query}':\n" +
                                     "\n".join([f"- {r.get('title')}: {r.get('snippet')}" for r in searchResults]))

        if additionalContext:
            enrichedContext = "\n\nAdditional information from web search:\n" + "\n---\n".join(additionalContext)
            state.ragContext = (state.ragContext or "") + enrichedContext
            state.addMessage(speaker="system", content=f"Enriched context with {len(additionalContext)} web searches.")

        return state

class LLMNode(GZNode):
    def __init__(self, nodeName: str, toolRegistry: Optional[Any] = defaultToolRegistry):
        super().__init__(nodeName)
        self.toolRegistry = toolRegistry

    async def streamExecute(self, state: GZAgentState, config: Optional[Dict] = None) -> AsyncGenerator[Dict[str, Any], GZAgentState]:
        prompt = self.constructPrompt(state)
        modelIdToUse = state.modelId or Constants.DEFAULT_MODEL_ID

        yield {"type": "llm_prompt", "nodeName": self.nodeName, "data": {"promptLength": len(prompt), "modelId": modelIdToUse}}

        currentAiResponseContent = ""
        inputTokens = 0
        outputTokens = 0

        try:
            responseStream = interactWithBedrock(prompt, modelIdToUse)
            if responseStream is None:
                state.errorOccurred = True
                state.errorMessage = "LLMNode error: Failed to get response stream from Bedrock."
                state.addMessage(speaker="system", content=state.errorMessage, isError=True)
                yield {"type": "error", "nodeName": self.nodeName, "detail": state.errorMessage}
                return

            for event in responseStream:
                chunk = event.get('chunk')
                if chunk:
                    chunkData = json.loads(chunk.get('bytes').decode())
                    chunkType = chunkData.get('type')

                    if chunkType == 'content_block_delta':
                        delta = chunkData.get('delta', {})
                        text = delta.get('text', '')
                        if text:
                            currentAiResponseContent += text
                            yield {"type": "llm_chunk", "text": text}
                    elif chunkType == 'message_delta':
                        usage = chunkData.get('usage', {})
                        if 'output_tokens' in usage:
                            outputTokens = usage['output_tokens']
                    elif chunkType == 'message_stop':
                        bedrockMetadata = chunkData.get('amazon-bedrock-invocationMetrics', {})
                        inputTokens = bedrockMetadata.get('inputTokenCount', 0)
                        outputTokens = bedrockMetadata.get('outputTokenCount', outputTokens or 0)
                        break
                elif any(errKey in event for errKey in ['internalServerException', 'modelStreamErrorException', 'validationException', 'throttlingException', 'modelTimeoutException']):
                    errorKey = next(iter(event))
                    errorMessageContent = event[errorKey].get('message', 'Unknown LLM streaming error')
                    state.errorOccurred = True
                    state.errorMessage = f"LLMNode error: {errorKey} - {errorMessageContent}"
                    yield {"type": "error", "nodeName": self.nodeName, "detail": state.errorMessage}
                    state.addMessage(speaker="system", content=state.errorMessage, isError=True)
                    return
        except Exception as e:
            state.errorOccurred = True
            state.errorMessage = f"LLMNode streaming exception: {str(e)}"
            yield {"type": "error", "nodeName": self.nodeName, "detail": state.errorMessage}
            state.addMessage(speaker="system", content=state.errorMessage, isError=True)
            return

        parsedToolCalls: List[GZToolCallRequest] = self.parseLlmResponseForToolCalls(currentAiResponseContent)

        finalContentForUser = currentAiResponseContent
        serializable_tool_calls_for_event_and_message: Optional[List[Dict[str, Any]]] = None

        if parsedToolCalls:
            state.requestedToolCalls.extend(parsedToolCalls)

            serializable_tool_calls_for_event_and_message = [tc.model_dump() for tc in parsedToolCalls]
            yield {"type": "tool_call_request", "nodeName": self.nodeName, "data": serializable_tool_calls_for_event_and_message}
            finalContentForUser = "(Decided to use tools)"

        state.addMessage(speaker="ai", content=finalContentForUser, toolCalls=serializable_tool_calls_for_event_and_message)

        state.internalScratchpad['llmInputTokens'] = inputTokens
        state.internalScratchpad['llmOutputTokens'] = outputTokens

        state.currentStepOutput = {"rawLlmResponse": currentAiResponseContent, "parsedToolCalls": serializable_tool_calls_for_event_and_message}

        yield {"type": "node_complete", "nodeName": self.nodeName, "data": state.currentStepOutput}
        state.currentStepOutput = None
        return

    def constructPrompt(self, state: GZAgentState, config: Optional[Dict] = None) -> str:
        history = state.getFormattedHistoryForPrompt()
        toolsJsonSchema = self.toolRegistry.getAvailableToolsSchemas() if self.toolRegistry else []
        systemPromptLines = ["You are a helpful AI assistant."]
        reactInstructions = """
        Before providing a tool call or a final answer, you MUST first output your reasoning and plan in a <thought> XML block.
        The <thought> block should detail your analysis of the user's request, what information you have, what information you need, and if you plan to use a tool, which one and why.
        After the </thought> block, you will EITHER provide a JSON object for tool calls OR a direct natural language answer to the user.

        Example 1: Using a tool
        <thought>
        The user is asking for the current weather in London. I don't have this information. I should use the 'getWeather' tool. The location is London.
        </thought>
        {
        "tool_calls": [
            {
            "tool_name": "getWeather",
            "tool_input": { "location": "London" }
            }
        ]
        }

        Example 2: Answering directly after a tool provided information
        <thought>
        The user asked for the current date. The 'getCurrentDateTime' tool has provided the date as 2025-05-07 and the day as Wednesday. I have sufficient information to answer the user's request directly.
        </thought>
        Today is Wednesday, May 7th, 2025.

        Follow this thought process strictly. The <thought> block always comes first, followed by your action (tool call or direct answer).
        """
        systemPromptLines.append(reactInstructions)
        if state.executedToolResults:
            systemPromptLines.append(
                "\nYou have just received results from one or more tools."
                " Use these tool results, along with the conversation history, to directly answer the user's most recent question or fulfill their request."
                " Do NOT call the same tool again for the same purpose if the information has been provided unless the results were unsatisfactory or incomplete."
                " Only call tools if new, distinct information is needed to answer the user's request or if a previous tool call failed."
                " Synthesize the information into a natural language response."
            )

        toolInstructions = ""
        if toolsJsonSchema:
            toolInstructions = f"""
            You have access to the following tools. If you need to use a tool to answer the user's request, respond with a JSON object in the following format EXACTLY:
            {{
            "tool_calls": [
                {{
                "tool_name": "tool_name_here",
                "tool_input": {{ "arg1": "value1", "arg2": "value2" ... }}
                }}
            ]
            }}
            If you need to use multiple tools, include multiple tool call objects in the "tool_calls" list.
            Only respond with this JSON structure if you intend to call a tool. Otherwise, respond directly to the user in natural language.

            Available Tools:
            {json.dumps(toolsJsonSchema, indent=2)}
            """
            systemPromptLines.append(toolInstructions)

        # Add the standard RAG context
        if state.ragContext and (not config or not config.get("disableRag", False)):
            systemPromptLines.append(f"\nUse the following RAG context to inform your answer if relevant:\n{state.ragContext}")

        # Add agentic RAG metadata if available
        retrievalPlan = state.internalScratchpad.get("retrievalPlan", "")
        contextAnalysis = state.internalScratchpad.get("contextAnalysis", "")

        if retrievalPlan or contextAnalysis:
            systemPromptLines.append("\n### INFORMATION RETRIEVAL ANALYSIS ###")

            if retrievalPlan:
                systemPromptLines.append(f"""
                Search strategy used:
                {retrievalPlan[:300]}...
                """)

            if contextAnalysis:
                systemPromptLines.append(f"""
                Analysis of retrieved information:
                {contextAnalysis[:300]}...
                """)

            systemPromptLines.append("""
            When responding, integrate this retrieval analysis in your thought process.
            Consider the quality and relevance of the retrieved information.
            """)

        finalSystemPrompt = "\n".join(systemPromptLines)

        lastUserMessageContent = state.getLastUserMessageContent()
        if not lastUserMessageContent and state.messages:
            for msg in reversed(state.messages):
                if msg.speaker == "user" and msg.contentType == "text":
                    lastUserMessageContent = msg.content
                    break
        if not lastUserMessageContent:
            lastUserMessageContent = state.userInput

        promptTemplate = f"""{finalSystemPrompt}

        Conversation History (including any recent tool responses):
        {history}

        Human: {lastUserMessageContent}
        Assistant:"""
        return promptTemplate.strip()


    def parseLlmResponseForToolCalls(self, llmResponse: str) -> List[Dict[str, Any]]:
        toolCalls = []
        try:
            jsonStart = llmResponse.find('{')
            jsonEnd = llmResponse.rfind('}') + 1
            if jsonStart != -1 and jsonEnd > jsonStart:
                potentialJson = llmResponse[jsonStart:jsonEnd]
                parsedJson = json.loads(potentialJson)
                if isinstance(parsedJson, dict) and "tool_calls" in parsedJson:
                    calls = parsedJson["tool_calls"]
                    if isinstance(calls, list):
                        for call in calls:
                            if isinstance(call, dict) and "tool_name" in call and "tool_input" in call:
                                toolCalls.append({
                                    "toolCallId": f"llmcall_{str(uuid.uuid4())}",
                                    "toolName": call["tool_name"],
                                    "toolInput": call["tool_input"]
                                })
                        return [GZToolCallRequest(**tc) for tc in toolCalls]
        except json.JSONDecodeError:
            pass
        return []


    async def execute(self, state: GZAgentState, config: Optional[Dict] = None) -> GZAgentState:
        async for _ in self.streamExecute(state, config):
            pass
        return state


class ToolExecutorNode(GZNode):
    def __init__(self, nodeName: str, toolRegistry: Any = defaultToolRegistry):
        super().__init__(nodeName)
        self.toolRegistry = toolRegistry

    async def execute(self, state: GZAgentState, config: Optional[Dict] = None) -> AsyncGenerator[Dict[str, Any], None]:
        if not state.requestedToolCalls:
            state.currentStepOutput = {"message": "No tools requested."}
            yield {"type": "node_complete", "nodeName": self.nodeName, "data": state.currentStepOutput}
            return

        executedResultsOutput = []
        for toolCallRequest in state.requestedToolCalls:
            tool = self.toolRegistry.getTool(toolCallRequest.toolName)
            if not tool:
                errorMsg = f"Tool '{toolCallRequest.toolName}' not found in registry."
                state.addToolCallResult(toolCallRequest.toolCallId, toolCallRequest.toolName, errorMsg, isError=True, errorMessage=errorMsg)
                state.addMessage(speaker="system", content=errorMsg, isError=True, toolCallId=toolCallRequest.toolCallId, toolName=toolCallRequest.toolName)
                executedResultsOutput.append({"toolCallId": toolCallRequest.toolCallId, "status": "error", "detail": errorMsg})
                continue

            try:
                validatedInput = tool.inputSchema(**toolCallRequest.toolInput)
                yield {"type": "tool_execution_start", "nodeName": self.nodeName, "data": {"toolName": tool.toolName, "toolInput": validatedInput.model_dump()}}

                toolOutput = await tool.executeTool(validatedInput)

                state.addToolCallResult(toolCallRequest.toolCallId, tool.toolName, toolOutput)
                state.addMessage(speaker="tool", content=toolOutput, toolCallId=toolCallRequest.toolCallId, toolName=tool.toolName)
                yield {"type": "tool_execution_result", "nodeName": self.nodeName, "data": {"toolName": tool.toolName, "toolOutput": toolOutput}}
                executedResultsOutput.append({"toolCallId": toolCallRequest.toolCallId, "status": "success"})

            except ValidationError as ve:
                errorMsg = f"Input validation error for tool '{tool.toolName}': {ve.errors()}"
                state.addToolCallResult(toolCallRequest.toolCallId, tool.toolName, errorMsg, isError=True, errorMessage=errorMsg)
                state.addMessage(speaker="system", content=errorMsg, isError=True, toolCallId=toolCallRequest.toolCallId, toolName=tool.toolName)
                yield {"type": "error", "nodeName": self.nodeName, "detail": errorMsg}
                executedResultsOutput.append({"toolCallId": toolCallRequest.toolCallId, "status": "validation_error", "detail": errorMsg})
            except Exception as e:
                errorMsg = f"Error executing tool '{tool.toolName}': {str(e)}"
                state.addToolCallResult(toolCallRequest.toolCallId, tool.toolName, errorMsg, isError=True, errorMessage=errorMsg)
                state.addMessage(speaker="system", content=errorMsg, isError=True, toolCallId=toolCallRequest.toolCallId, toolName=tool.toolName)
                yield {"type": "error", "nodeName": self.nodeName, "detail": errorMsg}
                executedResultsOutput.append({"toolCallId": toolCallRequest.toolCallId, "status": "execution_error", "detail": errorMsg})

        state.requestedToolCalls = []
        state.currentStepOutput = {"executedToolResults": executedResultsOutput}
        yield {"type": "node_complete", "nodeName": self.nodeName, "data": state.currentStepOutput}
    async def streamExecute(self, state: GZAgentState, config: Optional[Dict] = None) -> AsyncGenerator[Dict[str, Any], GZAgentState]:
        if not state.requestedToolCalls:
            state.currentStepOutput = {"message": "No tools requested."}
            yield {"type": "node_complete", "nodeName": self.nodeName, "data": state.currentStepOutput}
            return

        originalRequestedToolCalls = list(state.requestedToolCalls)
        state.requestedToolCalls = []

        for toolCallRequest in originalRequestedToolCalls:
            tool = self.toolRegistry.getTool(toolCallRequest.toolName)
            if not tool:
                errorMsg = f"Tool '{toolCallRequest.toolName}' not found in registry."
                state.addToolCallResult(toolCallRequest.toolCallId, toolCallRequest.toolName, errorMsg, isError=True, errorMessage=errorMsg)
                state.addMessage(speaker="system", content=errorMsg, isError=True, toolCallId=toolCallRequest.toolCallId, toolName=toolCallRequest.toolName)
                yield {"type": "tool_execution_result", "nodeName": self.nodeName, "data": {"toolName": toolCallRequest.toolName, "toolCallId": toolCallRequest.toolCallId, "status": "error", "output": errorMsg}}
                continue

            try:
                validatedInput = tool.inputSchema(**toolCallRequest.toolInput)
                yield {"type": "tool_execution_start", "nodeName": self.nodeName, "data": {"toolName": tool.toolName, "toolCallId": toolCallRequest.toolCallId, "toolInput": validatedInput.model_dump()}}

                toolOutput = await tool.executeTool(validatedInput)

                state.addToolCallResult(toolCallRequest.toolCallId, tool.toolName, toolOutput)
                state.addMessage(speaker="tool", content=toolOutput, toolCallId=toolCallRequest.toolCallId, toolName=tool.toolName)
                yield {"type": "tool_execution_result", "nodeName": self.nodeName, "data": {"toolName": tool.toolName, "toolCallId": toolCallRequest.toolCallId, "status": "success", "output": toolOutput}}

            except ValidationError as ve:
                errorMsg = f"Input validation error for tool '{tool.toolName}': {ve.errors()}"
                state.addToolCallResult(toolCallRequest.toolCallId, tool.toolName, errorMsg, isError=True, errorMessage=errorMsg)
                state.addMessage(speaker="system", content=errorMsg, isError=True, toolCallId=toolCallRequest.toolCallId, toolName=tool.toolName)
                yield {"type": "tool_execution_result", "nodeName": self.nodeName, "data": {"toolName": tool.toolName, "toolCallId": toolCallRequest.toolCallId, "status": "validation_error", "output": errorMsg}}
            except Exception as e:
                errorMsg = f"Error executing tool '{tool.toolName}': {str(e)}"
                state.addToolCallResult(toolCallRequest.toolCallId, tool.toolName, errorMsg, isError=True, errorMessage=errorMsg)
                state.addMessage(speaker="system", content=errorMsg, isError=True, toolCallId=toolCallRequest.toolCallId, toolName=tool.toolName)
                yield {"type": "tool_execution_result", "nodeName": self.nodeName, "data": {"toolName": tool.toolName, "toolCallId": toolCallRequest.toolCallId, "status": "execution_error", "output": errorMsg}}

        state.currentStepOutput = {"message": "Tool execution phase completed."}
        yield {"type": "node_complete", "nodeName": self.nodeName, "data": state.currentStepOutput}
        state.currentStepOutput = None
        return


class FinalOutputNode(GZNode):
    async def execute(self, state: GZAgentState, config: Optional[Dict] = None) -> GZAgentState:
        if not state.errorOccurred:
            aiMessages = [msg for msg in state.messages if msg.speaker == "ai" and not msg.toolCalls]
            if aiMessages:
                state.finalOutput = aiMessages[-1].content
            elif any(msg.speaker == "ai" and msg.toolCalls for msg in state.messages):
                 state.finalOutput = "(The assistant decided to use tools but did not provide a final verbal response to the user yet. Waiting for next step or user input.)"
            else:
                state.finalOutput = "No direct AI response was generated for the user in this step."
        else:
            state.finalOutput = state.errorMessage or "An error occurred during processing."
        state.currentStepOutput = {"finalAgentResponse": state.finalOutput}
        return state
