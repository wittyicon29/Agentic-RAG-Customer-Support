import streamlit as st
import os
import tempfile
import requests
from typing import List
from dotenv import load_dotenv

from agno.agent import Agent
from agno.document import Document
from agno.models.google import Gemini
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response

from agentic_rag import get_jiopay_support_agent

load_dotenv()

st.set_page_config(
    page_title="JioPay Support Assistant",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for app styling
CUSTOM_CSS = """
    <style>
    /* Dark Theme Styles */
    body {
        background-color: #1E1E1E;
        color: #ffffff;
    }

    /* Main Titles */
    .main-title {
        text-align: center;
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        padding: 1em 0;
    }
    .subtitle {
        text-align: center;
        color: #B0B0B0;
        margin-bottom: 2em;
    }

    /* Common button styling */
    .stButton button, div[data-testid="stDownloadButton"] button {
        width: 100%;
        border-radius: 25px;
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: none;
        transition: all 0.3s ease;
        padding: 10px;
        font-size: 16px;
        cursor: pointer;
    }

    /* Consistent glow effect for all buttons */
    .stButton button:hover, div[data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.6);
        background-color: rgba(255, 255, 255, 0.2);
    }

    /* Chat Messages */
    .chat-container {
        border-radius: 15px;
        padding: 1em;
        margin: 1em 0;
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* Tool Calls */
    .tool-result {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 1em;
        margin: 1em 0;
        border-left: 4px solid #3B82F6;
    }

    /* Status Messages */
    .status-message {
        padding: 1em;
        border-radius: 10px;
        margin: 1em 0;
    }
    .success-message {
        background-color: #155724;
        color: #d4edda;
    }
    .error-message {
        background-color: #721c24;
        color: #f8d7da;
    }

    /* Sidebar Styling */
    .sidebar .stButton button {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }

    /* Ensure Hover Effects Apply in Dark Mode */
    @media (prefers-color-scheme: dark) {
        .chat-container, .tool-result {
            background-color: rgba(255, 255, 255, 0.15);
        }
    }
    </style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "jiopay_agent" not in st.session_state:
    st.session_state["jiopay_agent"] = None
if "loaded_urls" not in st.session_state:
    st.session_state.loaded_urls = set()
if "knowledge_base_initialized" not in st.session_state:
    st.session_state.knowledge_base_initialized = False


def restart_agent():
    """Reset the agent and clear chat history"""
    logger.debug("---*--- Restarting agent ---*---")
    st.session_state["jiopay_agent"] = None
    st.session_state["messages"] = []
    st.session_state.knowledge_base_initialized = False
    st.rerun()


def add_message(role, content, tool_calls=None):
    """Add a message to the chat history"""
    message = {"role": role, "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    st.session_state["messages"].append(message)
    return message


def display_tool_calls(container, tool_calls):
    """Display tool calls in the UI"""
    if not tool_calls:
        return
    
    tool_calls_html = ""
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get("name", "Unknown Tool")
        tool_input = tool_call.get("input", {})
        tool_output = tool_call.get("output", "No output available")
        
        tool_calls_html += f"""
        <div class="tool-call">
            <div class="tool-call-header">🔧 {tool_name}</div>
            <div><strong>Input:</strong> {tool_input}</div>
            <div><strong>Output:</strong> {tool_output}</div>
        </div>
        """
    
    container.markdown(tool_calls_html, unsafe_allow_html=True)


def export_chat_history():
    """Export chat history to markdown format"""
    md_content = "# JioPay Support Assistant Chat History\n\n"
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            md_content += f"## User\n{content}\n\n"
        elif role == "assistant":
            md_content += f"## JioPay Support Assistant\n{content}\n\n"
            
            # Add tool calls if they exist
            if "tool_calls" in msg and msg["tool_calls"]:
                md_content += "### Tool Calls\n"
                for tc in msg["tool_calls"]:
                    md_content += f"- **{tc.get('name', 'Tool')}**\n"
                    md_content += f"  - Input: {tc.get('input', {})}\n"
                    md_content += f"  - Output: {tc.get('output', 'No output')}\n"
                md_content += "\n"
    
    return md_content


def about_widget():
    """Display about information in the sidebar"""
    with st.sidebar.expander("ℹ️ About JioPay Support Assistant"):
        st.markdown("""
        ### JioPay Support Assistant
        
        This AI-powered assistant provides comprehensive support for JioPay's digital payment platform, helping customers with:
        
        - Troubleshooting payment issues
        - Account management queries
        - Security concerns
        - Transaction processing
        - JioPay features and services
        
        The assistant uses retrieval-augmented generation to provide accurate information from JioPay's knowledge base and documentation.
        
        **Built with:**
        - [Streamlit](https://streamlit.io/)
        - [Agno AI Framework](https://github.com/agno-ai)
        - [Google Gemini](https://ai.google.dev/)
        - [LangChain](https://python.langchain.com/)
        """)


def initialize_agent(debug_mode=False, show_tool_calls=True):
    """Initialize or retrieve the JioPay Support Agent"""
    if "jiopay_agent" not in st.session_state or st.session_state["jiopay_agent"] is None:
        logger.info("---*--- Creating JioPay Support Agent ---*---")
        agent = get_jiopay_support_agent(
            debug_mode=debug_mode,
            show_tool_calls=show_tool_calls
        )
        st.session_state["jiopay_agent"] = agent
    return st.session_state["jiopay_agent"]


def main():
    ####################################################################
    # App header
    ####################################################################
    st.markdown("<h1 class='main-title'>JioPay Support Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Your intelligent customer service representative for JioPay's digital payment platform</p>",
        unsafe_allow_html=True,
    )

    ####################################################################
    # Initialize Agent
    ####################################################################
    jiopay_agent: Agent = initialize_agent(debug_mode=False, show_tool_calls=True)

    ####################################################################
    # Sample Questions
    ####################################################################
    st.sidebar.markdown("#### ❓ Sample Questions")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.sidebar.button("📝 Summarize Conversation", use_container_width=True):
            add_message(
                "user",
                "Can you summarize the chat?"
            )

    if st.sidebar.button("💰 Payment Issues", use_container_width=True):
        add_message(
            "user",
            "My payment through JioPay failed. What should I do?"
        )
    
    if st.sidebar.button("🏪 Merchant Services", use_container_width=True):
        add_message(
            "user",
            "How can I set up JioPay for my business?"
        )

    ####################################################################
    # Utility buttons
    ####################################################################
    st.sidebar.markdown("#### 🛠️ Utilities")
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            restart_agent()
    with col2:
        if st.download_button(
            "💾 Export Chat",
            export_chat_history(),
            file_name="jiopay_chat_history.md",
            mime="text/markdown",
            use_container_width=True,
        ):
            st.sidebar.success("Chat history exported!")

    ####################################################################
    # About section
    ####################################################################
    about_widget()

    ####################################################################
    # Display chat history
    ####################################################################
    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            _content = message["content"]
            if _content is not None:
                with st.chat_message(message["role"]):
                    # Display tool calls if they exist in the message
                    if "tool_calls" in message and message["tool_calls"]:
                        display_tool_calls(st.empty(), message["tool_calls"])
                    st.markdown(_content)

    ####################################################################
    # Handle user input
    ####################################################################
    if prompt := st.chat_input("Ask me anything about JioPay..."):
        user_message = add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
    ####################################################################
    # Generate response for user message
    ####################################################################
    last_message = (
        st.session_state["messages"][-1] if st.session_state["messages"] else None
    )
    if last_message and last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            # Create container for tool calls
            tool_calls_container = st.empty()
            resp_container = st.empty()
            with st.spinner("🤔 Thinking..."):
                response = ""
                try:
                    # Run the agent and stream the response
                    run_response = jiopay_agent.run(question, stream=True)
                    for _resp_chunk in run_response:
                        # Display tool calls if available
                        if hasattr(_resp_chunk, 'tools') and _resp_chunk.tools and len(_resp_chunk.tools) > 0:
                            display_tool_calls(tool_calls_container, _resp_chunk.tools)

                        # Display response
                        if hasattr(_resp_chunk, 'content') and _resp_chunk.content is not None:
                            response += _resp_chunk.content
                            resp_container.markdown(response)

                    # Add the complete response to message history
                    tools = None
                    if hasattr(jiopay_agent, 'run_response') and hasattr(jiopay_agent.run_response, 'tools'):
                        tools = jiopay_agent.run_response.tools
                    add_message("assistant", response, tools)
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    add_message("assistant", error_message)
                    st.error(error_message)


if __name__ == "__main__":
    main()