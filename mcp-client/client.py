import asyncio
import os
from socket import timeout
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from mcp_use import MCPAgent, MCPClient

async def main():
    # Load environment variables
    load_dotenv()

    # Create MCPClient from config file
    client = MCPClient.from_config_file(
        os.path.join(os.path.dirname(__file__), "browser_mcp.json")
    )

    # Create LLM
    # llm = ChatOpenAI(model="gpt-4o")
    # Alternative models:
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    # llm = ChatGroq(model="llama3-8b-8192")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    while True:
        try:
            query = input("Enter your query (or 'exit' to quit): ")
            if query.lower() in ["exit", "quit", "q", "e", "stop"]:
                break
            # Run the query
            result = await agent.run(
                query=query,
                max_steps=30,
            )
            print(f"\nResult: {result}")

        except Exception as e:
            print(f"An error occurred: {e}")
            if isinstance(e, timeout):
                print("Request timed out. Retrying...")
            else:
                break

if __name__ == "__main__":
    asyncio.run(main())