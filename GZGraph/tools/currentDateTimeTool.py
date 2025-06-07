import datetime
from GZGraph.gzToolBase import GZTool, GZToolInputSchema
from pydantic import Field

class CurrentDateTimeInput(GZToolInputSchema):
    timezone: str = Field(default="UTC", description="Optional timezone, e.g., 'America/New_York'. Defaults to UTC.")

class CurrentDateTimeTool(GZTool):
    def __init__(self):
        super().__init__(
            toolName="getCurrentDateTime",
            description="Gets the current date and time, optionally for a specific timezone.",
            inputSchema=CurrentDateTimeInput
        )

    async def executeTool(self, validatedInput: CurrentDateTimeInput, **kwargs) -> dict:
        try:
            now_local = datetime.datetime.now()
            now_utc = datetime.datetime.utcnow()
            return {
                "local_date": now_local.strftime("%Y-%m-%d"),
                "local_day": now_local.strftime("%A"),
                "local_time": now_local.strftime("%H:%M:%S"),
                "utc_date": now_utc.strftime("%Y-%m-%d"),
                "utc_day": now_utc.strftime("%A"),
                "utc_time": now_utc.strftime("%H:%M:%S UTC")
            }
        except Exception as e:
            return {"error": str(e)}