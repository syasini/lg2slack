"""Utility functions for Slack integration.

Helper functions for message formatting, markdown conversion, and event detection.
"""

import re
from typing import List, Dict


def is_bot_mention(text: str, bot_user_id: str) -> bool:
    """Check if bot is mentioned in a message.

    Slack mentions look like: <@U123ABC456>

    Args:
        text: Message text from Slack
        bot_user_id: Bot's Slack user ID (e.g., 'U123ABC456')

    Returns:
        True if bot is mentioned in the text

    Example:
        >>> is_bot_mention("<@U123> hello!", "U123")
        True
        >>> is_bot_mention("hello world", "U123")
        False
    """
    # Pattern matches: <@USER_ID>
    pattern = rf"<@{bot_user_id}>"
    return bool(re.search(pattern, text))


def is_dm(event: dict) -> bool:
    """Check if message is a direct message.

    Args:
        event: Slack event dict

    Returns:
        True if this is a direct message (DM)

    Example:
        >>> is_dm({"channel_type": "im"})
        True
        >>> is_dm({"channel_type": "channel"})
        False
    """
    return event.get("channel_type") == "im"


def clean_markdown(text: str) -> str:
    """Convert standard markdown to Slack mrkdwn format.

    Slack uses a slightly different markdown syntax (mrkdwn):
    - Links: [text](url) -> <url|text>
    - Images: ![alt](url) -> !<url|alt>
    - Bold: **text** -> *text*
    - Italic: *text* -> _text_
    - Code blocks: ```language -> ```

    Args:
        text: Standard markdown text

    Returns:
        Slack mrkdwn formatted text

    Example:
        >>> clean_markdown("Check [this link](https://example.com)")
        'Check <https://example.com|this link>'
    """
    # Convert markdown links: [text](url) -> <url|text>
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)

    # Convert markdown images: ![alt](url) -> !<url|alt>
    # Note: Slack doesn't render images inline in text, but this preserves the format
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"!<\2|\1>", text)

    # Convert bold: **text** -> *text* (Slack uses single asterisk for bold)
    text = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", text)

    # Convert italic: *text* -> _text_ (avoid matching already converted bold)
    # Negative lookbehind/ahead to avoid matching double asterisks
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"_\1_", text)

    # Clean up code blocks: remove language identifier after ```
    # Slack doesn't use language identifiers in the same way
    text = re.sub(r"^```[^\n]*\n", "```\n", text, flags=re.MULTILINE)

    # Convert bullet points: - or * at start of line -> •
    # text = re.sub(r"^\s*[-*]\s", "• ", text, flags=re.MULTILINE)

    return text


def extract_markdown_images(text: str) -> List[Dict]:
    """Extract markdown images and return Slack image blocks.

    Finds all markdown images in format: ![alt](url)
    Converts them to Slack image block format.

    Args:
        text: Text containing markdown images

    Returns:
        List of Slack image block dicts

    Example:
        >>> extract_markdown_images("Here's a chart: ![Sales Chart](https://example.com/chart.png)")
        [{'type': 'image', 'image_url': 'https://example.com/chart.png', 'alt_text': 'Sales Chart'}]
    """
    import logging
    logger = logging.getLogger(__name__)

    # Pattern matches: ![alt text](url)
    # Group 1: alt text (can be empty)
    # Group 2: url
    pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
    matches = re.findall(pattern, text)

    logger.info(f"Searching for markdown images in text (length={len(text)})")
    logger.debug(f"Text to search: {text}")
    logger.info(f"Found {len(matches)} markdown image patterns")

    image_blocks = []
    for alt_text, url in matches:
        block = {
            "type": "image",
            "image_url": url,
            "alt_text": alt_text or "Image",  # Default alt text if empty
        }
        logger.info(f"Creating image block: alt_text='{alt_text}', url='{url}'")
        image_blocks.append(block)

    return image_blocks


def create_feedback_block(
    thread_id: str = None,
    show_feedback_buttons: bool = True,
    show_thread_id: bool = True,
) -> List[Dict]:
    """Create Slack feedback button blocks with optional thread ID.

    Creates a context_actions block with feedback_buttons widget.
    Optionally includes a context block with thread_id for reporting.

    Args:
        thread_id: Optional LangGraph thread ID to display for reporting
        show_feedback_buttons: Whether to show feedback buttons (default: True)
        show_thread_id: Whether to show thread_id in footer (default: True)

    Returns:
        List of Slack block dicts with feedback buttons (and optional context)

    Example usage in Slack API:
        blocks = create_feedback_block(thread_id="abc-123")
        client.chat_postMessage(channel=channel, text=text, blocks=blocks)
    """
    blocks = []

    # Add context block with thread_id if provided and enabled
    if thread_id and show_thread_id:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"_Thread ID: `{thread_id}`_"
                }
            ]
        })

    # Add feedback buttons if enabled
    if show_feedback_buttons:
        blocks.append({
            "type": "context_actions",
            "elements": [
                {
                    "type": "feedback_buttons",
                    "action_id": "feedback",  # Used to identify button clicks
                    "positive_button": {
                        "text": {"type": "plain_text", "text": "Good Response"},
                        "value": "positive",
                    },
                    "negative_button": {
                        "text": {"type": "plain_text", "text": "Bad Response"},
                        "value": "negative",
                    },
                }
            ],
        })

    return blocks
