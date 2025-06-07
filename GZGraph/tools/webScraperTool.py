import httpx
from bs4 import BeautifulSoup
from typing import Dict
from pydantic import Field

from GZGraph.gzToolBase import GZTool, GZToolInputSchema

class WebScraperInput(GZToolInputSchema):
    url: str = Field(description="The URL of the website to scrape content from.")

class AsyncWebScraperTool(GZTool):
    def __init__(self):
        super().__init__(
            toolName="scrapeWebsite",
            description="Fetches and extracts the textual content from a given URL. Use this after a web search provides a promising URL, to get detailed information from that page.",
            inputSchema=WebScraperInput
        )
        self.httpClient = httpx.AsyncClient(timeout=25.0)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

    async def executeTool(self, validatedInput: WebScraperInput, **kwargs) -> Dict[str, str]:
        websiteUrl = validatedInput.url
        try:
            response = await self.httpClient.get(websiteUrl, headers=self.headers, follow_redirects=True)
            response.raise_for_status()
            
            if 'text/html' not in response.headers.get('content-type', '').lower():
                return {websiteUrl: f"Content is not HTML, but {response.headers.get('content-type')}. Scraper focused on text."}

            soup = BeautifulSoup(response.content, 'html.parser')
            
            for scriptOrStyle in soup(["script", "style", "header", "footer", "nav", "aside"]):
                scriptOrStyle.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            cleanText = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
            
            maxLength = kwargs.get("maxLength", 5000) 
            return {websiteUrl: cleanText[:maxLength] + ("..." if len(cleanText) > maxLength else "")}
        except httpx.HTTPStatusError as e:
            return {websiteUrl: f"Failed to retrieve content (HTTP {e.response.status_code}): {e.response.text if e.response else str(e)}"} # Added more detail
        except httpx.RequestError as e:
            return {websiteUrl: f"Failed to retrieve content (Request Error): {e}"}
        except Exception as e:
            return {websiteUrl: f"Unexpected error scraping {websiteUrl}: {str(e)}"}