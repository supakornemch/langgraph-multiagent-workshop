from multi_db_agents import app
from langchain_core.messages import HumanMessage
import sys

def test_workshop():
    config = {"configurable": {"thread_id": "workshop-v2"}}
    
    scenarios = [
        "How many PatchPanel-Y do we have in Thailand?",
        "What is the procedure for maintaining PatchPanels in TH?",
        "Our Database-Z in Singapore has high CPU, what should I check?",
        "Is there any stock of Firewall-X in UK?"
    ]
    
    print("=== LangGraph Multi-Agent Workshop Test (V2) ===\n")
    
    for query in scenarios:
        print(f"USER: {query}")
        # Use stream to show node transitions
        for chunk in app.stream({"messages": [HumanMessage(content=query)]}, config):
            for node, output in chunk.items():
                if "messages" in output:
                    last_msg = output["messages"][-1]
                    if last_msg.type == "ai" and last_msg.content:
                        # Clean up formatting for display
                        print(f"[{node.upper()}]: {last_msg.content.strip()}")
        print("-" * 50)

if __name__ == "__main__":
    test_workshop()
