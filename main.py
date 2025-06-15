# Basic imports 
from __future__ import annotations
import asyncio
import copy
import json
import logging
import os
from typing import Any, Callable, List, Optional
import inspect

# Loading Environment variables
from dotenv import load_dotenv
import os
load_dotenv()

# Importing Firecrawl 
from firecrawl import FirecrawlApp, ScrapeOptions

# Importing livekit 
from livekit.agents import Agent, AgentSession, JobContext, RunContext, WorkerOptions, cli, function_tool
from livekit.plugins import assemblyai, silero, google

# Importing Pydantic AI for MCP Server - Supabase
from pydantic_ai.mcp import MCPServerStdio

# Setting up logging
logging.basicConfig(level=logging.INFO) # Keeping it at INFO level 
logger = logging.getLogger(__name__) # Logger for this module

# Instantiating Firecrawl
firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Creating function tool for firecrawl which does a REST API call for scraping which later will be provided to Livekit agent.
@function_tool
async def firecrawl_websearch(
    context: RunContext, # Context from Livekit runtime
    query: str, # user query for web search
    limit: int = 5 # Maximum pages to crawl
    ) -> List[str]: 
    
    # Base URL - appending user query to google search
    url = f"https://www.google.com/search?q={query}"
    logger.info(f"Starting web search for query: {query} with limit: {limit}")

    # Asynchronously scraping the web using Firecrawl
    loop = asyncio.get_event_loop() 

    try:
        # Running the Firecrawl scrape method in an executor
        result = await loop.run_in_executor(
            None, 
            lambda: firecrawl_app.crawl_url(
                url, 
                limit=limit, 
                scrape_options=ScrapeOptions(formats=["text", "markdown"])))
        
        logger.info("Firecrawl returned %d pages", len(result))

        return result 
    
    except Exception as e:
        logger.error("Firecrawl search failed: %s", e, exc_info=True)
        return []

# Function to convert JSON schema types to Python typing annotations
def _py_type(schema: dict) -> Any:
    """Convert JSON schema types into Python typing annotations."""
    t = schema.get("type")
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
    }

    if isinstance(t, list):
        if "array" in t:
            return List[_py_type(schema.get("items", {}))]
        t = t[0]

    if isinstance(t, str) and t in mapping:
        return mapping[t]
    if t == "array":
        return List[_py_type(schema.get("items", {}))]

    return Any

# Function to convert JSON schema to Google-style docstring
def schema_to_google_docstring(description: str, schema: dict) -> str:
    """
    Generate a Google-style docstring section from a JSON schema.
    """
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [description or "", "Args:"]

    for name, prop in props.items():
        t = prop.get("type", "Any")
        if isinstance(t, list):
            if "array" in t:
                subtype = prop.get("items", {}).get("type", "Any")
                py_type = f"List[{subtype.capitalize()}]"
            else:
                py_type = t[0].capitalize()
        elif t == "array":
            subtype = prop.get("items", {}).get("type", "Any")
            py_type = f"List[{subtype.capitalize()}]"
        else:
            py_type = t.capitalize()

        if name not in required:
            py_type = f"Optional[{py_type}]"

        desc = prop.get("description", "")
        lines.append(f"    {name} ({py_type}): {desc}")

    return "\n".join(lines)

# Building Livekit tools from Supabase MCP server and storing in `tools` list which will be used later by Livekit agent.
async def build_tools_from_supabase_mcp_sever(supabase_mcp_server: MCPServerStdio) -> List[Callable]: 
    """
    This function builds tools from the Supabase MCP server.
    It uses the Pydantic AI MCP server to fetch tools and returns them.
    """

    # List to hold the tools
    tools: List[Callable] = []

    # Instanting the MCP server with Supabase token
   
    supabase_tools = await supabase_mcp_server.list_tools() # Fetching available tools from the MCP server
    logger.info("Fetched %d tools from Supabase MCP server", len(supabase_tools))

    # Converting the tools to Livekit compatible format
    for td in supabase_tools:
            if td.name == "deploy_edge_function":
                logger.warning("Skipping tool %s", td.name)
                continue

            schema = copy.deepcopy(td.parameters_json_schema)
            if td.name == "list_tables":
                props = schema.setdefault("properties", {})
                props["schemas"] = {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "default": []
                }
                schema["required"] = [r for r in schema.get("required", []) if r != "schemas"]

            props = schema.get("properties", {})
            required = set(schema.get("required", []))

            def make_proxy(
                tool_def=td,
                _props=props,
                _required=required,
                _schema=schema
            ) -> Callable:
                async def proxy(context: RunContext, **kwargs):
                    # Convert None → [] for array params
                    for k, v in list(kwargs.items()):
                        if ((_props[k].get("type") == "array"
                            or "array" in (_props[k].get("type") or []))
                                and v is None):
                            kwargs[k] = []

                    response = await supabase_mcp_server.call_tool(tool_def.name, arguments=kwargs or None)
                    if isinstance(response, list):
                        return response
                    if hasattr(response, "content") and response.content:
                        text = response.content[0].text
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return text
                    return response

                # Build signature from schema
                params = [
                    inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext)
                ]
                ann = {"context": RunContext}

                for name, ps in _props.items():
                    default = ps.get("default", inspect._empty if name in required else None)
                    params.append(
                        inspect.Parameter(
                            name,
                            inspect.Parameter.KEYWORD_ONLY,
                            annotation=_py_type(ps),
                            default=default,
                        )
                    )
                    ann[name] = _py_type(ps)

                proxy.__signature__ = inspect.Signature(params)
                proxy.__annotations__ = ann
                proxy.__name__ = tool_def.name
                proxy.__doc__ = schema_to_google_docstring(tool_def.description or "", _schema)
                return function_tool(proxy)

            tools.append(make_proxy())

    return tools

async def invoking_livekit_agent(ctx: JobContext) -> None:  
    """
    This function invokes the Livekit agent with the provided context.
    It sets up the agent with the necessary tools and runs it.
    """

    await ctx.connect() # Connecting to the Livekit job context

    supabase_mcp_server = MCPServerStdio(
        "npx",
        args=["-y", "@supabase/mcp-server-supabase@latest", "--access-token", os.getenv("SUPABASE_ACCESS_TOKEN")],
    )

    await supabase_mcp_server.__aenter__() # Entering the MCP server context

    # loading tools 
    try: 

        supabase_tools = await build_tools_from_supabase_mcp_sever(supabase_mcp_server) # Building tools from Supabase MCP server

        tools = supabase_tools + [firecrawl_websearch]  # Adding Firecrawl tool to the list

        # Defining livekit agent
        agent = Agent(
            instructions="""
            You are a helpful AI assistant. Based on the user query use either `firecrawl_websearch` tool to search the web for any latest data (like news, facts),
            or use tools from Supabase MCP server to answer the user query accordingly.
            """,
            tools=tools
        )

        # Running the agent session with the provided context
        session = AgentSession(
            vad=silero.VAD.load(min_silence_duration=0.1),
            stt=assemblyai.STT(),
            llm=google.beta.realtime.RealtimeModel(
                model="gemini-2.5-flash-preview-native-audio-dialog",
                voice="Puck",
                temperature=0.8
            )
        )

        await session.start(agent=agent, room=ctx.room)
        await session.generate_reply(instructions="Hello! How can I assist you today?")

        await session.start(agent=agent, room=ctx.room)
        await session.generate_reply(instructions="Hello! How can I assist you today?")

        # Keep the session alive until cancelled
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Session cancelled, shutting down.")

    finally:
        await supabase_mcp_server.__aexit__(None, None, None)


if __name__ == "__main__":

    # Trigger complete Livekit CLI app with the entrypoint function
    cli.run_app(WorkerOptions(entrypoint_fnc=invoking_livekit_agent))






