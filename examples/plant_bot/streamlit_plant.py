"""Streamlit app for testing plant agent with thread-based persistence."""

import streamlit as st
from plant_agent import graph_with_checkpointer as graph
from langchain_core.messages import HumanMessage

st.title("🌱 Houseplant Helper")
st.caption("Ask about plant care, recommendations, or request images!")

# Initialize session state for thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_session_1"

# Initialize chat history display
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about houseplants!"):
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream agent response with thread_id for automatic history
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Create config with thread_id
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Stream with "messages" mode for token-by-token streaming
        # This gives us incremental updates as the LLM generates tokens
        for message, metadata in graph.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config,
            stream_mode="messages"
        ):
            # Only process AI messages (type is "AIMessageChunk" during streaming)
            if hasattr(message, "type") and message.type == "AIMessageChunk":
                if hasattr(message, "content") and message.content:
                    content = message.content

                    # Handle both string and list content
                    if isinstance(content, str):
                        # Accumulate the content
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")

        # Final response without cursor
        response_placeholder.markdown(full_response)

    # Add to display history
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a button to clear conversation (start new thread)
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.thread_id = f"streamlit_session_{len(st.session_state.get('thread_id', ''))}"
    st.rerun()
