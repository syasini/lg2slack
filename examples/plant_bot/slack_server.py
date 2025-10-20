"""Slack bot server using lg2slack.

Just a few lines of code to connect the LangGraph agent to Slack!
"""

import re
from lg2slack import SlackBot

bot = SlackBot(
    streaming=True,
    reply_in_thread=True,
    show_feedback_buttons=True,
    show_thread_id=True,
    extract_images=True,
    include_metadata=True,
    enable_feedback_comments= True,
    processing_reaction="eyes",
    )

@bot.transform_input
async def talk_like_a_plant(message: str) -> str:
    """Transform user messages to sound like a plant."""
    return f"[Plant Voice: talk like a plant from the 80s!] {message}"

@bot.transform_output
async def remove_search_tags(message: str) -> str:
    """Remove content between <search> XML tags."""
    return re.sub(r'<search>.*?</search>', '', message, flags=re.DOTALL).strip()


@bot.transform_output
async def add_greeting(message: str, context) -> str:
    """Add a greeting with the user's name."""
    return f"Hello <@{context.user_id}>!\n\n{message}"


# Export the app for langgraph.json
app = bot.app
