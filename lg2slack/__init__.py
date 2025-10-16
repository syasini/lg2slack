"""lg2slack - Simple LangGraph to Slack integration.

This package provides an easy way to connect LangGraph applications to Slack
with minimal configuration. Just create a SlackBot, optionally add transformers,
and export the FastAPI app to langgraph.json.

Example:
    from lg2slack import SlackBot

    bot = SlackBot()

    @bot.transform_input
    async def add_context(message, context):
        return f"User {context.user_name}: {message}"

    app = bot.app  # Export to langgraph.json
"""

from .bot import SlackBot
from .config import BotConfig, MessageContext

__version__ = "0.1.0"
__all__ = ["SlackBot", "BotConfig", "MessageContext"]
