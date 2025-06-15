# MCP Powered Voice Agent

User will interact with the agent, using voice input, based on the user's transcript, agent will choose either Firecrawl or interact with Supabase MCP server and fetch the relevant details from the database. Livekit will act as an AI conversational bot.

## Tech Stack

- Assembly AI (speech-to-text conversion)

- Supabase MCP Server
- Firecrawl (for handling Web Search)
- Livekit (text-to-speech and conversational capability)
- Gemini 2.0 LLM

## Working in

1. User asks a query over the livekit bot, and it gets converted from speech-to-text using AssemblyAI.

2. The transcript gets passed to the AI Agent, which was already equipped with Firecrawl web search along with an Supabase MCP Server.

3. Based on user's request, Agent chooses the tool to interact with, either with the Supabase or Firecrawl and queries the same.
4. Once the agent receives the result, it gets passed to Livekit bot and gets voiced out. 
