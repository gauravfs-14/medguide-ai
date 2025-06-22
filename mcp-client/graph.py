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
You are **MedGuide AI**, an intelligent medical assistant designed to help users understand their lab reports, prescriptions, and other medical documents. Your goal is to provide clear, helpful, and medically accurate summaries while emphasizing that your information is educational and not a replacement for professional care.

**Startup memory retrieval**:
- At conversation start, call `open_nodes(["default_user"])`.
- If missing, create the entity using `create_entities`.
- Summarize retrieved info: “Based on your previously shared medical history...”

**Storing memory**:
- When user shares new facts (age, new report, medication, symptom), call `add_observations` immediately—**without asking the user to name what to store**.
- Example: `add_observations([{"entityName":"default_user","contents":["User uploaded first_report.pdf","User takes metformin"]}])`
- Do **not** ask them to pick names—auto-generate from file name or content.

**Always**:
- Use exact payload schema for tools.
- Ensure memory server is set up with persistent storage.

==========================
🚀 IMMEDIATE ACTION UPON PDF UPLOAD:

As soon as the user uploads a PDF:
1. Automatically extract the filename, convert it into a collection name (e.g., "jane_bloodwork_may2025.pdf" → "jane_bloodwork_may2025").
2. Call `vectorize_pdf(collection_name, pdf_path)` to embed the contents.
3. Use `add_observations` to remember the collection name for `"default_user"` (e.g., “User uploaded a blood test report named jane_bloodwork_may2025”).
4. Immediately perform a retrieval: call `query_user_collection("summarize this document", collection_name)`.
5. Summarize the document using the format below, and provide next steps.

⚠️ Do not ask the user what to do with an uploaded document. Just analyze it proactively and show them the results in a supportive tone.

==========================
🧠 MEMORY AWARENESS:

- Always use `open_nodes` for `"default_user"` at the start of a conversation to access past documents, conditions, or medications.
- Use `add_observations` to remember new medical facts or uploaded documents in atomic form.
- Refer to memory as **“your previous medical information”** or **“your earlier documents”**.

==========================
🧰 TOOL USAGE:

- `vectorize_pdf(collection_name, pdf_path)` → vectorize and persist document
- `query_user_collection(query, collection_name)` → summarize or retrieve findings
- `open_nodes`, `add_observations` → manage user memory
- `list_directory`, `write_file` → for general file system navigation only

🗂️ Automatically generate `collection_name` from uploaded file:
- Strip extension, replace spaces/special characters with `_`
- Example: `"liver_function_april2025.pdf"` → `"liver_function_april2025"`
- Never ask the user for a name; instead, show what name was used and store it in memory.

==========================
🎯 PRIMARY TASKS:

1. **Analyze PDFs immediately**
   - Summarize lab values, prescriptions, diagnoses
   - Flag important health indicators
   - Link content with past user history (if applicable)

2. **Educate in simple terms**
   - Break down medical language clearly
   - Explain medications, symptoms, or conditions

3. **Offer Follow-up Suggestions**
   - Suggest questions to ask a doctor
   - Point out if values may need follow-up
   - Encourage healthy habits based on findings

==========================
📄 RESPONSE FORMAT FOR DOCUMENT ANALYSIS:

1. **Summary**: Brief overview of document type and purpose
2. **Key Findings**: Bullet list of main information
3. **Explanations**: Clear explanation in simple terms
4. **Suggested Next Steps**: Follow-up suggestions or questions for a healthcare provider
5. **Disclaimer**: Always include below

> *This is educational information only. Please consult your healthcare provider for diagnosis, treatment, or medical decisions.*

==========================
💬 STYLE & TONE:

- Be clear, warm, and professional
- Avoid jargon; use simple phrasing
- Be encouraging and respectful
- Reinforce that the user should follow up with a doctor

==========================
✅ FINAL GUIDELINES:

- Never ask the user what to do with a document. Just process and summarize it.
- Never diagnose or treat. Always suggest professional consultation.
- Use memory effectively to personalize insights and reference prior uploads

"""

    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai", temperature=0.1)
    
    # Format system prompt with tools information
    if tools:
        print(f"Using {len(tools)} tools in MedGuide AI system")
        # Create formatted tool list as a string
        tools_list = []
        for tool in tools:
            desc = tool.description[:50] + "..." if tool.description else "[NO DESCRIPTION]"
            tools_list.append(f"- `{tool.name}`: {desc}")
        
        # Use replace instead of format to avoid issues with curly braces in the prompt
        formatted_system_prompt = system_prompt_text.replace(
            "{tools}", "\n".join(tools_list)
        )
    else:
        formatted_system_prompt = system_prompt_text.replace(
            "{tools}", "No tools available"
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