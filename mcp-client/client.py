import streamlit as st
import asyncio
import json
import time
import os
import tempfile
from typing import Dict, Union
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient

from graph import build_agent_graph

# Load environment variables
load_dotenv(override=True)

# Get config file path - use relative path or environment variable
CONFIG_FILE = os.getenv("MCP_FILE_PATH", "mcp_config.json")
if not os.path.isabs(CONFIG_FILE):
    CONFIG_FILE = os.path.join(os.path.dirname(__file__), CONFIG_FILE)

# Load MCP server configuration
try:
    with open(CONFIG_FILE, 'r') as f:
        MCP_SERVER_CONFIG = json.load(f)
except FileNotFoundError:
    st.error(f"❌ MCP configuration file not found: {CONFIG_FILE}")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"❌ Invalid JSON in MCP configuration: {e}")
    st.stop()

print(f"🔧 Using MCP configuration from: {MCP_SERVER_CONFIG}")

# --- Session State Initialization ---
def init_session_state():
    """Initialize all session state variables"""
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "tools" not in st.session_state:
        st.session_state.tools = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "message_counter" not in st.session_state:
        st.session_state.message_counter = 0

# --- LangGraph Initialization ---
async def init_graph():
    try:
        # Convert the MCP config format to what MultiServerMCPClient expects
        connections = {}
        
        if "mcpServers" in MCP_SERVER_CONFIG:
            # Handle the format with "mcpServers" wrapper
            for server_name, server_config in MCP_SERVER_CONFIG["mcpServers"].items():
                # Convert to the expected format
                connection_config = {
                    "command": server_config["command"],
                    "args": server_config["args"],
                    "transport": server_config.get("transport", "stdio")  # Always include transport
                }
                
                connections[server_name] = connection_config
        else:
            # Direct format without wrapper
            connections = MCP_SERVER_CONFIG
        
        print(f"🔧 Connecting to MCP servers: {list(connections.keys())}")
        print(f"🔧 Connection configs: {connections}")
        
        client = MultiServerMCPClient(connections=connections)
        tools = await client.get_tools()
        
        # Deduplicate tools by name
        unique_tools = {tool.name: tool for tool in tools}
        tools = list(unique_tools.values())
        
        if not tools:
            raise ValueError("No tools found in MCP servers")
            
        print(f"✅ Successfully loaded {len(tools)} tools from MCP servers")
        
        st.session_state.tools = tools
        st.session_state.graph = build_agent_graph(tools=tools)
        return True
    except Exception as e:
        print(f"❌ MCP initialization error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error(f"❌ Failed to initialize MCP tools: {str(e)}")
        return False

# --- Message Management ---
def add_message(role: str, content: str, msg_type: str = "text", metadata: Union[Dict, None] = None):
    """Add a message to chat history with proper structure"""
    if not content:
        return
    
    st.session_state.message_counter += 1
    message = {
        "id": st.session_state.message_counter,
        "role": role,
        "content": content,  # Store content as-is without stripping
        "type": msg_type,
        "timestamp": time.time(),
        "metadata": metadata or {}
    }
    st.session_state.chat_history.append(message)

# --- Chat Rendering ---
def render_chat_history():
    """Render the complete chat history with proper formatting"""
    for msg in st.session_state.chat_history:
        render_single_message(msg)

def render_single_message(msg: Dict):
    """Render a single message with appropriate styling"""
    with st.chat_message(msg["role"]):
        if msg["type"] == "reasoning":
            st.markdown("🧠 **Agent Reasoning**")
            with st.expander("💭 View thought process", expanded=False):
                st.markdown(msg["content"])
        elif msg["type"] == "tool_call":
            st.markdown("🛠️ **Tool Call**")
            with st.expander(f"📋 Calling `{msg['metadata'].get('tool_name', 'unknown')}`", expanded=False):
                if msg["metadata"].get("args"):
                    st.code(json.dumps(msg["metadata"]["args"], indent=2), language="json")
                else:
                    st.markdown(msg["content"])
        elif msg["type"] == "tool_response":
            st.markdown("📊 **Tool Response**")
            with st.expander(f"📋 Result from `{msg['metadata'].get('tool_name', 'unknown')}`", expanded=False):
                try:
                    # Try to format as JSON if possible
                    content = msg["content"]
                    if content.strip().startswith(('{', '[')):
                        parsed = json.loads(content)
                        st.code(json.dumps(parsed, indent=2), language="json")
                    else:
                        st.markdown(content)
                except json.JSONDecodeError:
                    st.markdown(msg["content"])
                except Exception:
                    st.markdown(msg["content"])
        else:
            # Regular AI response message - render content as-is to preserve formatting
            content = msg["content"]
            if content and content not in [None, ""]:
                # Only apply safe formatting for security, preserve all whitespace and structure
                formatted_content = safe_format_content(content)
                st.markdown(formatted_content, unsafe_allow_html=False)

# --- Content Formatting ---
def safe_format_content(content: str) -> str:
    """Safely format content while preserving markdown structure."""
    if not content or not isinstance(content, str):
        return ""
    # Only remove potentially harmful HTML, don't strip whitespace
    content = content.replace("<script>", "&lt;script&gt;")
    content = content.replace("</script>", "&lt;/script&gt;")
    return content

# --- Reasoning Parser ---
class ReasoningParser:
    def __init__(self):
        self.reasoning_buffer = ""
        self.in_reasoning = False
        self.reasoning_complete = False
    
    def process_content(self, content: str):
        """Process content and extract reasoning vs response parts"""
        result = {
            "reasoning_content": "",
            "response_content": "",
            "reasoning_started": False,
            "reasoning_ended": False
        }
        
        # Handle content with <think> tags
        if "<think>" in content:
            parts = content.split("<think>", 1)
            if len(parts) > 1:
                result["response_content"] = parts[0]  # Content before <think>
                reasoning_part = parts[1]
                
                # Check if reasoning ends in this chunk
                if "</think>" in reasoning_part:
                    think_parts = reasoning_part.split("</think>", 1)
                    result["reasoning_content"] = think_parts[0]
                    result["response_content"] += think_parts[1]  # Content after </think>
                    result["reasoning_ended"] = True
                else:
                    result["reasoning_content"] = reasoning_part
                
                result["reasoning_started"] = True
                self.in_reasoning = True
        
        elif "</think>" in content and self.in_reasoning:
            parts = content.split("</think>", 1)
            result["reasoning_content"] = parts[0]
            result["response_content"] = parts[1] if len(parts) > 1 else ""
            result["reasoning_ended"] = True
            self.in_reasoning = False
            self.reasoning_complete = True
        
        elif self.in_reasoning:
            result["reasoning_content"] = content
        
        else:
            result["response_content"] = content
        
        return result

# --- Streaming Handler ---
class StreamingHandler:
    def __init__(self):
        self.reasoning_container = None
        self.response_container = None
        self.reasoning_buffer = ""
        self.response_buffer = ""
        self.parser = ReasoningParser()
        self.reasoning_finalized = False
        self.has_response_content = False  # Track if we have actual response content

    def create_reasoning_display(self):
        with st.chat_message("assistant"):
            st.markdown("🧠 **Thinking...**")
            with st.expander("💭 Reasoning Process", expanded=True):
                return st.empty()

    def create_response_display(self):
        with st.chat_message("assistant"):
            return st.empty()

    def update_reasoning(self, content: str):
        self.reasoning_buffer += content
        if self.reasoning_container:
            # Don't format during streaming, just display raw content
            self.reasoning_container.markdown(self.reasoning_buffer)

    def update_response(self, content: str):
        self.response_buffer += content
        self.has_response_content = True
        if self.response_container:
            # Don't format during streaming, preserve all content including whitespace
            if self.response_buffer:
                self.response_container.markdown(self.response_buffer, unsafe_allow_html=False)

    def finalize_reasoning(self):
        if self.reasoning_buffer and not self.reasoning_finalized:
            self.reasoning_finalized = True
            if self.reasoning_container:
                # Only apply minimal formatting when finalizing
                self.reasoning_container.markdown(self.reasoning_buffer)
            return self.reasoning_buffer
        return ""

    def reset_for_new_reasoning(self):
        """Reset the handler to allow for a new reasoning block"""
        self.reasoning_container = None
        self.reasoning_buffer = ""
        self.reasoning_finalized = False
        self.has_response_content = False
        # Don't reset response_container and response_buffer if they have content
        if not self.has_response_content:
            self.response_container = None
            self.response_buffer = ""
        self.parser = ReasoningParser()  # Reset parser state

    def cleanup_empty_response_container(self):
        """Clean up empty response container to prevent UI issues"""
        if self.response_container and not self.has_response_content:
            self.response_container = None
            self.response_buffer = ""

    def has_pending_response(self):
        """Check if there's pending response content that should be saved"""
        return self.response_buffer and self.has_response_content

    def get_pending_response(self):
        """Get and clear pending response content"""
        if self.has_pending_response():
            # Return content without stripping to preserve formatting
            content = self.response_buffer
            self.response_buffer = ""
            self.response_container = None
            self.has_response_content = False
            return content
        return ""

    def process_chunk(self, content: str):
        parsed = self.parser.process_content(content)

        if parsed["reasoning_started"] and not self.reasoning_container:
            self.reasoning_container = self.create_reasoning_display()

        if parsed["reasoning_content"]:
            self.update_reasoning(parsed["reasoning_content"])

        if parsed["reasoning_ended"]:
            return self.finalize_reasoning()

        if parsed["response_content"]:
            # Only create response container if we have actual content
            if parsed["response_content"].strip():
                if not self.response_container:
                    self.response_container = self.create_response_display()
                self.update_response(parsed["response_content"])

        return None

# --- User Input Handler ---
async def handle_user_input(user_input: str):
    """Process user input and generate assistant response"""
    if st.session_state.is_processing:
        st.warning("Please wait for the current response to complete.")
        return
    
    # Add user message
    add_message("user", user_input)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.is_processing = True
    
    try:
        # Prepare for streaming
        config: RunnableConfig = {"configurable": {"thread_id": "1"}}
        messages = [HumanMessage(content=user_input)]
        graph = st.session_state.graph
        
        # Initialize streaming handler
        handler = StreamingHandler()
        
        # Stream the response
        async for msg, _ in graph.astream(
            {"messages": messages}, 
            config=config, 
            stream_mode="messages"
        ):
            if isinstance(msg, AIMessageChunk):
                content = str(msg.content or "")
                
                if content:
                    # Process the content chunk
                    finalized_reasoning = handler.process_chunk(content)
                    
                    # If reasoning was finalized, add to history
                    if finalized_reasoning:
                        add_message("assistant", finalized_reasoning, "reasoning")
                
                # Handle tool calls
                if msg.tool_calls:
                    # Save any pending response content before tool calls
                    pending_response = handler.get_pending_response()
                    if pending_response:
                        add_message("assistant", pending_response, "text")
                    
                    for call in msg.tool_calls:
                        tool_name = call.get("name", "unknown")
                        tool_args = call.get("args", {})
                        
                        # Display tool call
                        with st.chat_message("assistant"):
                            st.markdown("🛠️ **Tool Call**")
                            with st.expander(f"Calling `{tool_name}`", expanded=False):
                                st.code(json.dumps(tool_args, indent=2), language="json")
                        
                        # Add to history
                        add_message("assistant", f"Calling tool: {tool_name}", "tool_call", {
                            "tool_name": tool_name,
                            "args": tool_args
                        })
                    
                    # Clean up any empty response containers and reset for new reasoning
                    handler.cleanup_empty_response_container()
                    handler.reset_for_new_reasoning()
            
            elif isinstance(msg, ToolMessage):
                # Handle tool responses
                tool_name = getattr(msg, 'name', 'unknown')
                
                try:
                    # Process tool response content
                    if isinstance(msg.content, str):
                        content = msg.content
                    else:
                        content = json.dumps(msg.content, indent=2)
                    
                    # Display tool response
                    with st.chat_message("assistant"):
                        st.markdown("📋 **Tool Response**")
                        with st.expander(f"Response from `{tool_name}`", expanded=False):
                            if content.strip().startswith(('{', '[')):
                                try:
                                    parsed = json.loads(content)
                                    st.code(json.dumps(parsed, indent=2), language="json")
                                except:
                                    st.markdown(content)
                            else:
                                st.markdown(content)
                    
                    # Add to history
                    add_message("assistant", content, "tool_response", {
                        "tool_name": tool_name
                    })
                
                except Exception as e:
                    error_msg = f"Error processing tool response: {str(e)}"
                    st.error(error_msg)
                    add_message("assistant", error_msg, "tool_response", {
                        "tool_name": tool_name,
                        "error": True
                    })
                
                # Reset handler for potential new reasoning after tool response
                handler.reset_for_new_reasoning()
        
        # Finalize any remaining response content as separate message
        final_response = handler.get_pending_response()
        if final_response:
            add_message("assistant", final_response, "text")
    
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        add_message("assistant", f"Error: {str(e)}", "text")
    
    finally:
        st.session_state.is_processing = False

# --- Main UI ---
def main():
    # Page configuration
    st.set_page_config(
        page_title="🧠 MedGuide AI - Medical Assistant",
        page_icon="�",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("� MedGuide AI - Your Personal Medical Assistant")
    st.markdown("*AI-powered assistant for understanding health reports, medications, and medical documents*")
    
    # Initialize graph if needed
    if st.session_state.graph is None:
        with st.spinner("🔄 Initializing AI agent and MCP tools..."):
            try:
                # Use nest_asyncio to handle Streamlit's event loop
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                pass
            
            success = asyncio.run(init_graph())
            if success:
                st.rerun()
            else:
                st.error("❌ Failed to initialize assistant. Please check your MCP configuration.")
                return
    
    # Sidebar with info
    with st.sidebar:
        st.header("📋 About MedGuide AI")
        st.markdown("""
        **What I can help with:**
        - 🧪 Lab test results interpretation
        - 💊 Medication information and dosages  
        - 📄 Medical document analysis
        - 🔍 Health term explanations
        - 📊 Health data visualization
        """)
        
        # Show available tools
        if st.session_state.tools:
            st.header("🔧 Available Tools")
            tool_count = len(st.session_state.tools)
            st.markdown(f"**{tool_count} tools loaded:**")
            
            # Group tools by type for better display
            tool_types = {}
            for tool in st.session_state.tools:
                tool_type = "Other"
                if "file" in tool.name.lower():
                    tool_type = "📁 File Operations"
                elif "chroma" in tool.name.lower() or "vector" in tool.name.lower():
                    tool_type = "🔍 Search & RAG"
                elif "calendar" in tool.name.lower():
                    tool_type = "📅 Calendar"
                
                if tool_type not in tool_types:
                    tool_types[tool_type] = []
                tool_types[tool_type].append(tool.name)
            
            for tool_type, tools in tool_types.items():
                with st.expander(f"{tool_type} ({len(tools)})"):
                    for tool_name in tools[:5]:  # Show first 5
                        st.markdown(f"• `{tool_name}`")
                    if len(tools) > 5:
                        st.markdown(f"*...and {len(tools) - 5} more*")

        st.header("📁 Upload Medical Documents")
        st.markdown("Upload lab reports, prescriptions, or medical documents for analysis:")
        
        uploaded_file = st.file_uploader(
            label="Choose a file...",
            type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'docx'],
            help="Supported: PDF, Images (PNG/JPG), Text files, Word documents"
        )

        if uploaded_file is not None:
            # Save uploaded file to temp directory
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Save file path to use in next prompt
            st.session_state.uploaded_file_path = temp_path
            st.success(f"✅ Uploaded: `{uploaded_file.name}`")
            st.info("💡 Ask me about this file in the chat below!")

        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.message_counter = 0
            st.rerun()
        
        # Quick action buttons
        st.header("🚀 Quick Actions")
        if st.button("📋 Explain my lab results", use_container_width=True):
            if hasattr(st.session_state, 'uploaded_file_path'):
                prompt = "Please analyze my lab results and explain what they mean in simple terms."
                asyncio.run(handle_user_input(prompt))
                st.rerun()
            else:
                st.warning("Please upload a lab report first!")
        
        if st.button("💊 Check my medications", use_container_width=True):
            if hasattr(st.session_state, 'uploaded_file_path'):
                prompt = "Please review my medications and explain what they're for, dosages, and any important notes."
                asyncio.run(handle_user_input(prompt))
                st.rerun()
            else:
                st.warning("Please upload a prescription or medication list first!")

    # Chat display area
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            # Show welcome message when no chat history
            st.markdown("""
            ### 👋 Welcome to MedGuide AI!
            
            I'm here to help you understand your medical information. You can:
            
            - **Upload documents** using the sidebar (lab reports, prescriptions, etc.)
            - **Ask questions** about your health data
            - **Get explanations** of medical terms and procedures
            - **Understand** your test results and medications
            
            **Example questions:**
            - "What does my TSH level of 4.2 mean?"
            - "Explain my blood pressure medication"
            - "What should I know about this lab report?"
            
            *Note: I provide educational information only. Always consult your healthcare provider for medical advice.*
            """)
        else:
            render_chat_history()
    
    # Input area
    if st.session_state.is_processing:
        st.info("🤖 Analyzing your medical information...")
        st.chat_input("Please wait for the current analysis to complete...", disabled=True)
    else:
        prompt = st.chat_input("Ask me about your medical documents or health questions...")

        if prompt:
            # Check if a file was uploaded and not yet used
            file_path = st.session_state.pop("uploaded_file_path", None)
            if file_path:
                prompt += f'\n\n[UPLOADED FILE: {file_path}]'

            # Use improved async handling
            try:
                # Apply nest_asyncio for better Streamlit compatibility
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                pass
            
            asyncio.run(handle_user_input(prompt))
            st.rerun()

if __name__ == "__main__":
    main()