"""Houseplant recommendation agent with conditional image search.

Simple, fast agent that:
1. Responds immediately with plant care knowledge
2. Only searches for images when needed
3. Uses MessagesState with thread_id for automatic conversation history

This design ensures low latency for simple questions while providing
images when the user asks about specific plants.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
import re

load_dotenv()

# Define state - extends MessagesState for automatic message history
class PlantAgentState(MessagesState):
    """State with automatic message history tracking."""
    needs_search: bool
    search_query: str
    search_results: str

# Initialize LLM (Haiku 3.5 for speed)
llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.7, streaming=True)

# Initialize Tavily search
tavily = TavilySearch(
    max_results=3,
    include_images=True,
    search_depth="basic",
)

# System prompt
SYSTEM_PROMPT = """You are a helpful houseplant expert assistant. CRITICAL: Keep responses SHORT and CONCISE.

You help people with:
- Plant care (watering, light, soil, etc.)
- Plant recommendations for different conditions
- Showing images of plants

BREVITY RULES (MUST FOLLOW):
- Maximum 2-3 sentences per response
- Use bullet points for lists (max 3-4 items)
- NO long explanations or elaborations
- NO unnecessary details or background information
- Get straight to the point

If the user asks to see what a plant looks like, or asks about a specific plant they want to see:
- Start your response with: <search> plant_name </search>
- Then continue with your answer

If you have search results with images, include them using markdown: ![plant name](IMAGE_URL)

REMEMBER: Brief, direct answers only. Quality over quantity."""
 
def respond_node(state: PlantAgentState) -> dict:
    """Generate response, determine if search is needed."""
    search_results = state.get("search_results", "")

    # Build messages with system prompt
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Add message history (automatically tracked by MessagesState)
    messages.extend(state["messages"])

    # Add search results if available
    if search_results:
        messages.append(HumanMessage(content=f"Here are images and information about the plant:\n{search_results}\n\nNow provide your response with the images."))

    # Generate response
    response = llm.invoke(messages)

    # Check if we need to search by looking for <search> tags
    needs_search = False
    search_query = ""

    # Extract search query from response if <search> tags are present
    match = re.search(r"<search>(.*?)</search>", response.content)
    if match:
        needs_search = True
        search_query = match.group(1).strip()
        # Remove the <search> tags and query from response
        response.content = re.sub(r"<search>.*?</search>", "", response.content).strip()

    return {
        "messages": [response],
        "needs_search": needs_search,
        "search_query": search_query,
    }

def search_node(state: PlantAgentState) -> dict:
    """Search for plant images and information."""
    search_query = state.get("search_query", "")

    if not search_query:
        return {"search_results": ""}

    # Search for the plant
    full_query = f"{search_query} houseplant care images"
    results = tavily.invoke(full_query)

    return {
        "search_results": str(results),
        "needs_search": False,  # Reset for next iteration
    }

def should_search(state: PlantAgentState) -> str:
    """Conditional edge: decide if we need to search."""
    if state.get("needs_search", False):
        return "search"
    return "end"

# Build the graph
workflow = StateGraph(PlantAgentState)

# Add nodes
workflow.add_node("respond", respond_node)
workflow.add_node("search", search_node)

# Set entry point
workflow.set_entry_point("respond")

# Add conditional edge from respond
workflow.add_conditional_edges(
    "respond",
    should_search,
    {
        "search": "search",
        "end": END,
    }
)

# Search loops back to respond
workflow.add_edge("search", "respond")

# Compile graph - for LangGraph Platform (persistence handled automatically)
graph = workflow.compile()

# Also export a version with checkpointer for local testing (Streamlit)
checkpointer = MemorySaver()
graph_with_checkpointer = workflow.compile(checkpointer=checkpointer)
