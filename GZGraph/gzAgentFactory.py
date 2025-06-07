from GZGraph.gzGraph import GZGraph
from GZGraph.gzNodes import (
    InitialProcessingNode, RAGNode, LLMNode, ToolExecutorNode, FinalOutputNode,QueryPlanningNode,
    EnhancedRAGNode, ContextAnalysisNode, InformationEnrichmentNode
)
from GZGraph.gzState import GZAgentState
from GZGraph.gzToolRegistry import defaultToolRegistry

def createToolUsingAgent(useAgenticRAG: bool = False) -> GZGraph:
    agent = GZGraph()

    initialNode = InitialProcessingNode("userInputProcessor")
    llmNode = LLMNode("languageModelProcessor", toolRegistry=defaultToolRegistry)
    toolNode = ToolExecutorNode("toolExecutor", toolRegistry=defaultToolRegistry)
    outputNode = FinalOutputNode("responseGenerator")

    agent.addNode(initialNode)
    agent.addNode(llmNode)
    agent.addNode(toolNode)
    agent.addNode(outputNode)

    if useAgenticRAG:
        queryPlanningNode = QueryPlanningNode("queryPlanner")
        ragNode = EnhancedRAGNode("contextRetriever")
        contextAnalysisNode = ContextAnalysisNode("contextAnalyzer")
        infoEnrichmentNode = InformationEnrichmentNode("infoEnricher")

        agent.addNode(queryPlanningNode)
        agent.addNode(ragNode)
        agent.addNode(contextAnalysisNode)
        agent.addNode(infoEnrichmentNode)

        agent.setEntryPoint(initialNode.nodeName)

        agent.addEdge(initialNode.nodeName, queryPlanningNode.nodeName)
        agent.addEdge(queryPlanningNode.nodeName, ragNode.nodeName)
        agent.addEdge(ragNode.nodeName, contextAnalysisNode.nodeName)

        def routeAfterAnalysis(state: GZAgentState) -> str:
            if state.internalScratchpad.get("needsMoreInfo", False):
                return infoEnrichmentNode.nodeName
            return llmNode.nodeName

        agent.addConditionalRouter(contextAnalysisNode.nodeName, routeAfterAnalysis)
        agent.addEdge(infoEnrichmentNode.nodeName, llmNode.nodeName)
    else:
        ragNode = RAGNode("contextRetriever")
        agent.addNode(ragNode)
        agent.setEntryPoint(initialNode.nodeName)
        agent.addEdge(initialNode.nodeName, ragNode.nodeName)
        agent.addEdge(ragNode.nodeName, llmNode.nodeName)

    def routeAfterLlm(state: GZAgentState) -> str:
        if state.errorOccurred:
            return outputNode.nodeName
        if state.requestedToolCalls:
            return toolNode.nodeName
        return outputNode.nodeName

    agent.addConditionalRouter(llmNode.nodeName, routeAfterLlm)
    agent.addEdge(toolNode.nodeName, llmNode.nodeName)
    agent.addEdge(outputNode.nodeName, agent.finishPoint)

    return agent
