import asyncio
from typing import Dict, List
from pydantic import Field
from duckduckgo_search import DDGS

from GZGraph.gzToolBase import GZTool, GZToolInputSchema

class DDGSearchInput(GZToolInputSchema):
    query: str = Field(description="The search query to find information on the web.")
    numResults: int = Field(default=10, description="Number of search result snippets to return.", ge=1, le=25)

class AsyncDDGSearchTool(GZTool):
    def __init__(self):
        super().__init__(
            toolName="webSearch",
            description="Searches the web for information based on a query using DuckDuckGo and returns a list of search result snippets. Use this to find general information, current events, or leads to specific websites.",
            inputSchema=DDGSearchInput
        )

    async def executeTool(self, validatedInput: DDGSearchInput, **kwargs) -> List[Dict[str, str]]:
        query = validatedInput.query
        num_results = validatedInput.numResults

        try:
            def blocking_ddg_search():
                with DDGS(timeout=20) as ddgs:
                    return ddgs.text(
                        keywords=query,
                        max_results=num_results
                    )

            ddg_results = await asyncio.to_thread(blocking_ddg_search)

            formatted_results = []
            if ddg_results:
                for res in ddg_results:
                    formatted_results.append({
                        "title": res.get("title", "No Title"),
                        "link": res.get("href", ""),
                        "snippet": res.get("body", "No snippet available.")
                    })
            
            return formatted_results if formatted_results else [{"message": "No relevant search results found via DuckDuckGo."}]

        except Exception as e:
            return [{"error": f"Unexpected error during DuckDuckGo web search: {str(e)}"}]