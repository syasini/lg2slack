"""lg2slack - Simple LangGraph to Slack integration.

⚠️  DEPRECATION WARNING ⚠️
This package has been renamed to 'langgraph2slack'.
Please update your dependencies:

    uv pip uninstall lg2slack
    uv pip install langgraph2slack

And update your imports:

    from langgraph2slack import SlackBot  # instead of lg2slack

This package will continue to work but will not receive updates.
All future development happens in the 'langgraph2slack' package.

---

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

import warnings

# Issue deprecation warning on import
warnings.warn(
    "lg2slack has been renamed to langgraph2slack. "
    "Please update your dependencies: uv pip install langgraph2slack. "
    "This package will not receive further updates.",
    DeprecationWarning,
    stacklevel=2
)

from .bot import SlackBot
from .config import BotConfig, MessageContext

# Version is dynamically loaded from package metadata (pyproject.toml)
# Use `uv version --bump patch|minor|major` to update version
try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("lg2slack")
except PackageNotFoundError:
    # Fallback for development (package not installed)
    __version__ = "0.0.0.dev"

__all__ = ["SlackBot", "BotConfig", "MessageContext"]
