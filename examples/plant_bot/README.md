# Houseplant Helper Example

A houseplant recommendation bot that demonstrates lg2slack integration with conditional tool routing.

## Features

- 🌱 Expert advice on houseplant care
- 🔍 Web search (only when images needed)
- 🖼️ Plant images on demand
- ⚡ Fast streaming responses with low latency
- 💬 Works in Slack DMs and channels
- 📝 Automatic conversation history with thread-based persistence

## Quick Start

### 1. Install Dependencies

```bash
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e ".[dev]"
```

### 2. Set Up Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required keys:
- `ANTHROPIC_API_KEY` - Get from https://console.anthropic.com
- `TAVILY_API_KEY` - Get from https://tavily.com
- `SLACK_BOT_TOKEN` - Get from https://api.slack.com/apps
- `SLACK_SIGNING_SECRET` - Get from https://api.slack.com/apps
- `ASSISTANT_ID` - Set to `plant_agent` (the graph name)
- `LANGGRAPH_URL` - Set to `http://localhost:2024` for local dev

### 3. Test Locally with Streamlit

```bash
streamlit run streamlit_plant.py --server.port 8503
```

Open http://localhost:8503 and ask questions like:
- "What are some easy houseplants for beginners?"
- "Show me what a monstera deliciosa looks like"
- "How much light does a pothos need?"
- "I want a low-light plant for my bathroom"

### 4. Deploy to Slack

#### Local Development

1. Start LangGraph dev server:
   ```bash
   langgraph dev
   ```

2. In another terminal, start ngrok:
   ```bash
   ngrok http 2024
   ```

3. Update your Slack app's Event Subscriptions URL to:
   ```
   https://your-ngrok-url.ngrok.io/events/slack
   ```

4. Test in Slack by DMing the bot or @mentioning it in a channel!

#### Production Deployment

1. Deploy to LangGraph Platform:
   ```bash
   langgraph deploy
   ```

2. Update your Slack app's Event Subscriptions URL to:
   ```
   https://your-deployment.langgraph.app/events/slack
   ```

3. Update `.env` to remove `LANGGRAPH_URL` (uses loopback on platform)

## How It Works

### Architecture

```
User Message → Claude Response → [Conditional] → Tavily Search → Claude Response (with images)
                                      ↓
                                    [End]
```

The agent uses **conditional routing** for optimal performance:
1. Receives user question about houseplants
2. Claude generates initial response (fast!)
3. **Conditional edge**: If user wants images, route to search node
4. Tavily searches for plant images (only when needed)
5. Claude generates final response with images
6. Streams all responses for low latency

This design keeps simple questions fast (no unnecessary search) while providing images on demand.

### Files

- `plant_agent.py` - LangGraph agent with conditional search routing
  - Exports `graph` (for LangGraph Platform, no checkpointer)
  - Exports `graph_with_checkpointer` (for local testing with persistence)
- `slack_server.py` - lg2slack integration (just 4 lines!)
- `streamlit_plant.py` - Local testing UI with thread-based conversation history
- `langgraph.json` - LangGraph Platform configuration
- `.env` - Environment variables (not in git)

### Key Concepts

**Thread-based persistence**: The agent uses LangGraph's built-in `MessagesState` with thread IDs for automatic conversation history tracking. Pass a config with thread_id:

```python
config = {"configurable": {"thread_id": "user_123"}}
graph.stream({"messages": [HumanMessage(content=prompt)]}, config)
```

**Dual export pattern**: `plant_agent.py` exports two versions:
- `graph` - For LangGraph Platform (persistence handled automatically)
- `graph_with_checkpointer` - For local testing (uses `MemorySaver()`)

This allows the same agent code to work both locally and on the platform.

## Customization

### Change the Agent Behavior

Edit the system prompt in `plant_agent.py` to customize:
- Types of plant information provided
- Tone and style of responses
- When to trigger image searches (currently uses "NEED_SEARCH:" prefix)

### Add Conditional Routing

The agent demonstrates conditional edges with `should_search()`. Add more conditional logic by:
1. Adding state fields to `PlantAgentState`
2. Creating decision functions like `should_search()`
3. Using `add_conditional_edges()` in the workflow

### Customize Slack Integration

Use lg2slack transformers in `slack_server.py`:

```python
from lg2slack import SlackBot

bot = SlackBot()

@bot.transform_input
async def add_context(message, context):
    return f"User {context.user_id}: {message}"

@bot.transform_output
async def add_footer(response, context):
    return f"{response}\n\n_Powered by LangGraph_"

app = bot.app
```

## Troubleshooting

### Images Not Showing

The agent uses a "NEED_SEARCH:" prefix in responses to trigger image searches. Check that:
- The system prompt includes the NEED_SEARCH instruction
- The `respond_node` correctly parses the prefix
- Tavily API key is set correctly

### Slow Responses

The agent is optimized for speed:
- Claude Sonnet 3.7 (faster than Sonnet 4.5)
- Conditional search (only when needed, not for every query)
- Tavily set to `search_depth="basic"`
- Token-by-token streaming with `stream_mode="messages"`

### No Streaming in Streamlit

Make sure you're using:
- `stream_mode="messages"` (not "values")
- Checking for `message.type == "AIMessageChunk"` during streaming
- Accumulating tokens: `full_response += content` (not replacing)

### Conversation History Not Working

Check that:
- You're passing `config = {"configurable": {"thread_id": "..."}}`
- Using `graph_with_checkpointer` for local testing (not plain `graph`)
- Thread ID is consistent across conversation turns

### Slack Bot Not Responding

1. Check ngrok is running and forwarding to port 2024
2. Verify Slack Event Subscriptions URL is correct
3. Check `langgraph dev` logs for errors
4. Ensure bot is invited to the channel (for @mentions)
5. Verify `ASSISTANT_ID` in `.env` is set to `plant_agent`

## Learn More

- [lg2slack Documentation](../../README.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Persistence Guide](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Slack Bolt Documentation](https://slack.dev/bolt-python/)
