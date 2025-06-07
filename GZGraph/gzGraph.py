from typing import Dict, Callable, Optional, AsyncGenerator, Any
import uuid
from GZGraph.gzState import GZAgentState
from GZGraph.gzNodes import GZNode

class GZGraph:
    def __init__(self):
        self.nodes: Dict[str, GZNode] = {}
        self.edges: Dict[str, str] = {} 
        self.conditionalRouters: Dict[str, Callable[[GZAgentState], str]] = {}
        self.entryPoint: Optional[str] = None
        self.finishPoint: str = "END_OF_GRAPH"

    def addNode(self, node: GZNode) -> None:
        if node.nodeName in self.nodes:
            raise ValueError(f"Node with name '{node.nodeName}' already exists.")
        self.nodes[node.nodeName] = node

    def addEdge(self, fromNodeName: str, toNodeName: str) -> None:
        if fromNodeName not in self.nodes:
            raise ValueError(f"Source node '{fromNodeName}' for edge not defined.")
        if toNodeName != self.finishPoint and toNodeName not in self.nodes:
            raise ValueError(f"Target node '{toNodeName}' for edge not defined.")
        if fromNodeName in self.conditionalRouters:
            raise ValueError(f"Node '{fromNodeName}' already has a conditional router. Cannot add a static edge.")
        self.edges[fromNodeName] = toNodeName
    
    def addConditionalRouter(self, fromNodeName: str, routerFunction: Callable[[GZAgentState], str]) -> None:
        if fromNodeName not in self.nodes:
            raise ValueError(f"Source node '{fromNodeName}' for conditional router not defined.")
        if fromNodeName in self.edges:
            raise ValueError(f"Node '{fromNodeName}' already has a static edge. Cannot add a conditional router.")
        self.conditionalRouters[fromNodeName] = routerFunction

    def setEntryPoint(self, nodeName: str) -> None:
        if nodeName not in self.nodes:
            raise ValueError(f"Entry point node '{nodeName}' not defined.")
        self.entryPoint = nodeName

    def setFinishPoint(self, nodeName: str) -> None:
        self.finishPoint = nodeName

    async def invokeStream(self, initialState: GZAgentState, config: Optional[Dict] = None) -> AsyncGenerator[Dict[str, Any], None]:
            if not self.entryPoint:
                yield {"type": "graph_error", "detail": "Graph entry point not set."}
                return

            config = config or {}
            state = initialState
            currentNodeName = self.entryPoint
            maxSteps = config.get("maxSteps", 15)
            currentStep = 0
            continueOnError = config.get("continueOnError", False)
            continueOnErrorInNode = config.get("continueOnErrorInNode", False)

            graphRunId = f"gzrun_{str(uuid.uuid4())}"
            yield {"type": "graph_start", "runId": graphRunId, "initialStateSnapshot": state.model_dump(exclude_none=True)}

            while currentNodeName != self.finishPoint and currentStep < maxSteps:
                if state.errorOccurred and not continueOnError:
                    yield {"type": "graph_error", "nodeName": currentNodeName, "detail": state.errorMessage, "runId": graphRunId}
                    break

                if currentNodeName not in self.nodes:
                    yield {"type": "graph_error", "detail": f"Node '{currentNodeName}' not found.", "runId": graphRunId}
                    break

                nodeToExecute = self.nodes[currentNodeName]
                yield {"type": "node_start", "nodeName": nodeToExecute.nodeName, "step": currentStep, "runId": graphRunId}

                try:
                    async for update in nodeToExecute.streamExecute(state, config):
                        update["step"] = currentStep
                        update["runId"] = graphRunId
                        yield update

                        if update.get("type") == "error" and not continueOnErrorInNode:
                            state.errorOccurred = True
                            state.errorMessage = state.errorMessage or update.get("detail", "")
                            break 

                    if state.errorOccurred and not continueOnError:
                        break

                except Exception as e:
                    state.errorOccurred = True
                    state.errorMessage = f"Unhandled exception in node {nodeToExecute.nodeName}: {str(e)}"
                    yield {"type": "node_error_unhandled", "nodeName": nodeToExecute.nodeName, "detail": state.errorMessage, "step": currentStep, "runId": graphRunId}
                    if not continueOnError:
                        break

                yield {"type": "node_end", "nodeName": nodeToExecute.nodeName, "step": currentStep, "runId": graphRunId, "stateSnapshot": state.model_dump(exclude_none=True, exclude={"userInput"})} # Avoid redundant large fields
                currentStep += 1

                nextNodeName = None

                if currentNodeName in self.conditionalRouters:
                    router = self.conditionalRouters[currentNodeName]
                    try:
                        nextNodeName = router(state)
                    except Exception as e:
                        state.errorOccurred = True
                        state.errorMessage = f"Error in router for node {currentNodeName}: {str(e)}"
                        yield {"type": "graph_error", "detail": state.errorMessage, "runId": graphRunId}
                        break
                elif currentNodeName in self.edges:
                    nextNodeName = self.edges[currentNodeName]
                else:
                    yield {"type": "graph_info", "detail": f"No outgoing edge or router from {currentNodeName}, ending graph.", "runId": graphRunId}
                    break

                if not nextNodeName or nextNodeName == self.finishPoint:
                    yield {"type": "graph_info", "detail": f"Transitioning to finish point '{self.finishPoint}' from {currentNodeName}.", "runId": graphRunId}
                    break

                if nextNodeName not in self.nodes and nextNodeName != self.finishPoint:
                    yield {"type": "graph_error", "detail": f"Router for {currentNodeName} returned invalid next node '{nextNodeName}'.", "runId": graphRunId}
                    break

                yield {"type": "edge_traversal", "fromNode": currentNodeName, "toNode": nextNodeName, "runId": graphRunId}
                currentNodeName = nextNodeName

            if currentStep >= maxSteps and currentNodeName != self.finishPoint:
                yield {"type": "graph_error", "detail": "Max steps reached.", "runId": graphRunId}

            yield {"type": "graph_complete", "finalStateSnapshot": state.model_dump(exclude_none=True), "runId": graphRunId, "totalSteps": currentStep}