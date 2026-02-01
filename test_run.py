import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from main import app  # Import the compiled app from main.py
from dotenv import load_dotenv

load_dotenv()

def run_test():
    print("--- Starting Automated Workshop Test ---\n")
    
    config = {"configurable": {"thread_id": "test-1"}}
    
    # scenario 1: Generic greeting -> Triage
    print("User: Hello, I have a problem.")
    for chunk in app.stream(
        {"messages": [HumanMessage(content="Hello, I have a problem.")]},
        config=config,
    ):
        for node_name, output in chunk.items():
            print(f"\n--- Node: {node_name} ---")
            if "messages" in output:
                last_msg = output["messages"][-1]
                if last_msg.type == "ai":
                    print(f"Assistant: {last_msg.content}")

    print("\n" + "="*50 + "\n")

    # scenario 2: Database specific -> Triage should handoff to Database Specialist
    print("User: My SQL database in Singapore is very slow.")
    for chunk in app.stream(
        {"messages": [HumanMessage(content="My SQL database in Singapore is very slow.")]},
        config=config,
    ):
        for node_name, output in chunk.items():
            print(f"\n--- Node: {node_name} ---")
            # In a handoff, we might see the triage_agent return a Command
            # and then the specialist node being executed.
            if "messages" in output:
                last_msg = output["messages"][-1]
                if last_msg.type == "ai":
                    print(f"Assistant: {last_msg.content}")

    print("\n" + "="*50 + "\n")

    # scenario 3: Network specific -> Triage should handoff to Network Specialist
    print("User: Please check packet loss in Singapore.")
    for chunk in app.stream(
        {"messages": [HumanMessage(content="Please check packet loss in Singapore.")]},
        config=config,
    ):
        for node_name, output in chunk.items():
            print(f"\n--- Node: {node_name} ---")
            if "messages" in output:
                last_msg = output["messages"][-1]
                if last_msg.type == "ai":
                    print(f"Assistant: {last_msg.content}")

    print("\n" + "="*50 + "\n")

    # scenario 4: Follow up -> should go DIRECTLY to Network Specialist
    print("User: What are the results for Singapore?")
    for chunk in app.stream(
        {"messages": [HumanMessage(content="What are the results for Singapore?")]},
        config=config,
    ):
        for node_name, output in chunk.items():
            print(f"\n--- Node: {node_name} ---")
            if "messages" in output:
                last_msg = output["messages"][-1]
                if last_msg.type == "ai":
                    print(f"Assistant: {last_msg.content}")

if __name__ == "__main__":
    run_test()
