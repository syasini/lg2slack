# lg2slack

Simple, minimal package to connect LangGraph applications to Slack with just a few lines of code.

## Installation

```bash
pip install lg2slack
```

## Quick Start

### 1. Create a Slack App

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From a manifest"
3. Copy the contents of `slack_manifest.yaml` from this repo
4. Replace placeholder values:
   - `your-app-name` → Your app name
   - `your-deployment-url` → Your ngrok URL (local) or LangGraph Platform URL (production)
5. Install the app to your workspace
6. Copy the Bot Token and Signing Secret

### 2. Configure Environment Variables

Create a `.env` file:

```bash
# Slack credentials (from https://api.slack.com/apps -> Your App)
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret

# LangGraph configuration
ASSISTANT_ID=your-assistant-id

# Optional: specify LangGraph URL (omit for loopback on platform)
# LANGGRAPH_URL=http://localhost:8123
```

### 3. Create Your Bot

```python
# server.py
from lg2slack import SlackBot

bot = SlackBot()

# Export the app for langgraph.json
app = bot.app
```

That's it! Just 4 lines of code.

### 4. Configure LangGraph Deployment

Add to your `langgraph.json`:

```json
{
  "dependencies": ["lg2slack", "."],
  "graphs": {
    "agent": "./your_agent.py:graph"
  },
  "env": ".env",
  "http": {
    "/events/slack": "server:app"
  }
}
```

### 5. Deploy

```bash
# Deploy to LangGraph Platform
langgraph deploy

# Your bot is now live! Chat with it in Slack by:
# - Sending a DM to the bot
# - @mentioning the bot in a channel
```

## Advanced Usage

### With Transformers

Customize message processing with input/output transformers:

```python
from lg2slack import SlackBot

bot = SlackBot()

# Add user context to messages
@bot.transform_input
async def add_context(message, context):
    return f"User {context.user_id} in {context.channel_id}: {message}"

# Add footer to responses
@bot.transform_output
async def add_footer(response, context):
    return f"{response}\n\n_Powered by LangGraph_"

app = bot.app
```

### Disable Streaming

If you prefer non-streaming responses:

```python
bot = SlackBot(streaming=False)
```

### Multiple Transformers

Transformers are applied in order:

```python
@bot.transform_input
async def first_transform(message, context):
    return f"[1] {message}"

@bot.transform_input
async def second_transform(message, context):
    return f"[2] {message}"

# Input "hello" becomes: "[2] [1] hello"
```

## Local Development

### 1. Start LangGraph with your bot

```bash
langgraph dev
# This automatically runs on http://localhost:8123 and serves your custom routes
```

Note: You don't need to run a separate server! LangGraph dev automatically imports and mounts your FastAPI app from `langgraph.json`.

### 2. Expose with ngrok

Install ngrok if you haven't already:
```bash
# macOS
brew install ngrok

# Or download from https://ngrok.com/download
```

Start ngrok to expose your local server:
```bash
ngrok http 8123
```

This will output something like:
```
Forwarding  https://abc123.ngrok.io -> http://localhost:8123
```

**Tip:** You can view all requests in ngrok's web interface at http://localhost:4040

### 3. Update Slack App

Go to your Slack app settings → Event Subscriptions:
- Request URL: `https://abc123.ngrok.io/events/slack` (use YOUR ngrok URL)
- Slack will verify the URL - you should see a green checkmark

### 4. Test

Send a DM to your bot or @mention it in a channel! You'll see requests in both:
- LangGraph dev console
- ngrok web interface (http://localhost:4040)

## Configuration Options

### SlackBot Parameters

```python
SlackBot(
    assistant_id: str = None,           # LangGraph assistant ID (or from env)
    langgraph_url: str = None,          # LangGraph URL, None for loopback
    streaming: bool = True,             # Enable streaming (default: True)
    slack_bot_token: str = None,        # Override Slack token (or from env)
    slack_signing_secret: str = None,   # Override Slack secret (or from env)
)
```

### Environment Variables

- `SLACK_BOT_TOKEN` - Required: Bot token from Slack
- `SLACK_SIGNING_SECRET` - Required: Signing secret from Slack
- `ASSISTANT_ID` - Required: LangGraph assistant ID
- `LANGGRAPH_URL` - Optional: LangGraph deployment URL (None = loopback)

## How It Works

### Architecture

```
Slack → lg2slack → [INPUT] → LangGraph [Human Message]
                                    ↓
Slack ← lg2slack ← [OUTPUT] ← LangGraph [AI Message]
```

### Message Flow

1. User sends message in Slack (DM or @mention)
2. Slack sends event to `/events/slack` endpoint
3. lg2slack applies input transformers
4. Message sent to LangGraph as Human Message
5. LangGraph processes and generates AI Message
6. LangGraph streams response chunks
7. Each chunk immediately forwarded to Slack (low latency!)
8. lg2slack applies output transformers
9. Final message displayed in Slack

### Thread Management

lg2slack automatically manages conversation threads:
- Slack threads map to LangGraph threads using format: `slack_{channel}_{timestamp}`
- Same Slack thread always connects to same LangGraph conversation
- LangGraph maintains conversation history

## Features

### Streaming

By default, lg2slack uses **true low-latency streaming**:
- Each token from LangGraph is immediately sent to Slack
- No waiting for complete response
- Better user experience with instant feedback

### Feedback (Coming Soon)

Integration with LangSmith for collecting user feedback on bot responses.

## Examples

See the `examples/` directory:
- `basic.py` - Minimal setup
- `with_transformers.py` - Using input/output transformers

## Troubleshooting

### Bot doesn't respond

1. Check Slack app Event Subscriptions URL is correct
2. Verify bot is invited to the channel (for @mentions)
3. Check logs: `langgraph dev` shows request logs
4. Ensure environment variables are set correctly
5. Check ngrok is running and forwarding correctly

### "Missing required configuration"

Make sure `.env` file exists with all required variables:
- `SLACK_BOT_TOKEN`
- `SLACK_SIGNING_SECRET`
- `ASSISTANT_ID`

### Streaming not working

1. Verify LangGraph is returning messages in streaming mode
2. Check that `streaming=True` (default)
3. Ensure your LangGraph app has a `messages` state key

### Local dev: "Connection refused"

Make sure `langgraph dev` is running before starting ngrok.

### ngrok URL keeps changing

Free ngrok URLs change each time you restart. Options:
- Use the new URL each time in Slack app settings
- Get a ngrok account for a persistent subdomain
- For production, deploy to LangGraph Platform (stable URL)

## Requirements

- Python 3.10+
- LangGraph deployment with `messages` state key
- Slack workspace with bot permissions

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
