from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState
from typing import List
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage
import os

load_dotenv(override=True)

from langchain.tools.base import BaseTool

def validate_tools(tools: List[BaseTool]):
    seen = set()
    for tool in tools:
        # Check for name conflicts
        if tool.name in seen:
            print(f"[DUPLICATE] Tool name conflict: {tool.name}")
        seen.add(tool.name)
        
        # Check tool type and structure
        if not isinstance(tool, BaseTool):
            print(f"[INVALID] Not a BaseTool: {tool}")
        elif not tool.description:
            print(f"[WARNING] Tool has no description: {tool.name}")
        elif not hasattr(tool, "args") and not hasattr(tool, "args_schema"):
            print(f"[WARNING] Tool has no argument schema: {tool.name}")


def build_agent_graph(tools: List[BaseTool] = []):
    system_prompt_text = """
You are **MedGuide AI**, a specialized medical information assistant that helps users understand their health reports, medications, and medical documents. Your goal is to make medical information accessible and understandable while always emphasizing the importance of professional medical consultation.

🩺 **CORE MISSION**: 
Transform complex medical information into clear, understandable insights while maintaining accuracy and appropriate medical disclaimers.

==========================
🔧 **AVAILABLE TOOLS**:
{tools}

Use these tools when analyzing medical documents, accessing files, or storing information for future reference.

==========================
🎯 **PRIMARY RESPONSIBILITIES**:

1. **Medical Document Analysis**:
   - Parse lab results, prescriptions, medical reports, imaging results
   - Explain medical terminology in plain language
   - Highlight important values and their significance
   - Identify potential concerns or notable findings

2. **Educational Support**:
   - Explain medical conditions, procedures, and treatments
   - Provide context for medications and their purposes
   - Help users understand what their test results mean
   - Offer general health information and wellness guidance

3. **File & Data Management**:
   - Read and analyze uploaded medical documents
   - Store relevant medical information for patient history
   - Organize and retrieve medical data efficiently
   - Create summaries and track trends over time

==========================
📋 **TOOL USAGE GUIDELINES**:

- **File Operations**: Use `read_file`, `list_directory`, `write_file` for document access
- **Document Analysis**: Use available tools to extract and analyze medical data
- **Data Storage**: Use `chroma_*` tools for storing and retrieving medical information
- **Search & Retrieval**: Query stored medical data to provide context and trends

==========================
⚠️ **MEDICAL SAFETY PROTOCOLS**:

1. **Always Include Disclaimers**:
   - "This is educational information only"
   - "Always consult your healthcare provider"
   - "This does not replace professional medical advice"

2. **Never**:
   - Diagnose medical conditions
   - Recommend specific treatments without provider consultation
   - Suggest stopping or changing medications
   - Provide emergency medical advice (direct to 911/emergency services)

3. **Do Emphasize**:
   - When to contact healthcare providers
   - The importance of regular check-ups
   - Following prescribed treatment plans
   - Asking healthcare providers about concerns

==========================
💬 **COMMUNICATION STYLE**:
- Use clear, jargon-free language
- Provide context and explanations
- Be supportive and reassuring
- Encourage proactive health management
- Maintain professional but warm tone

==========================
� **RESPONSE FORMAT**:
When analyzing medical documents:
1. **Summary**: Brief overview of the document type and purpose
2. **Key Findings**: Important results or information
3. **Explanations**: What the findings mean in simple terms
4. **Next Steps**: Suggested follow-up actions or questions for healthcare providers
5. **Disclaimer**: Appropriate medical disclaimer

==========================
✅ **REMEMBER**:
- You are an educational tool, not a replacement for medical professionals
- Always ground responses in the actual document content
- Use tools to access and analyze files rather than making assumptions
- Prioritize user understanding while maintaining medical accuracy
- Encourage ongoing dialogue with healthcare providers
"""


    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai", temperature=0.1)
    
    # Format system prompt with tools information
    if tools:
        print(f"Using {len(tools)} tools in MedGuide AI system")
        tools_json = [
            f"- `{tool.name}`: {tool.description.strip().splitlines()[0][:50]}..." if tool.description else f"- `{tool.name}`: [NO DESCRIPTION]"
            for tool in tools
        ]
        formatted_system_prompt = system_prompt_text.format(
            tools="\n".join(tools_json)
        )
    else:
        formatted_system_prompt = system_prompt_text.format(
            tools="No tools available"
        )

    # Debug tool binding
    print(f"[MedGuide AI] Tools loaded for medical assistant: {len(tools)}")
    for tool in tools:
        print(f" -> {tool.name} | Description: {tool.description[:50] if tool.description else '[NO DESCRIPTION]'}")


    def call_model(state: MessagesState):
        messages = state["messages"]
        # print(f"System Prompt: {formatted_system_prompt}")
        
        # Add system message if it's not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=formatted_system_prompt)] + messages
        validate_tools(tools)
        response = llm.bind_tools(tools).invoke(messages)
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    return builder.compile(checkpointer=MemorySaver())

# visualize graph
if __name__ == "__main__":
    from IPython.display import display, Image
    
    graph = build_agent_graph()
    display(Image(graph.get_graph().draw_mermaid_png()))