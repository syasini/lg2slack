"""Slack bot server using lg2slack.

Just 4 lines of code to connect the LangGraph agent to Slack!
"""

from lg2slack import SlackBot

bot = SlackBot(
    streaming=True,
    reply_in_thread=True,
    show_feedback_buttons=True,
    show_thread_id=True,
    )

# Export the app for langgraph.json
app = bot.app
