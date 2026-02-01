import os
from typing import Literal, Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent, InjectedState
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from dotenv import load_dotenv

# Load env for OPENAI_API_KEY
load_dotenv()

# --- 1. Setup Model ---
# In a real world, you might use AzureOpenAI
model = ChatOpenAI(model="gpt-4o", temperature=0)

# --- 2. Define Tools for Specialists ---

@tool
def check_database_latency(region: str):
    """Checks the database latency for a specific region."""
    # Mock response
    return f"Database latency in {region} is 45ms (Normal)."

@tool
def check_network_packet_loss(region: str):
    """Checks packet loss in a specific region."""
    # Mock response
    return f"Packet loss in {region} is 0.01% (Stable)."

# --- 3. Define Handoff Tools ---
# These tools allow agents to transfer control to one another.

def create_handoff_tool(agent_name: str):
    """Creates a tool that transfers control to another agent."""
    
    @tool(f"transfer_to_{agent_name}")
    def handoff_tool():
        """Transfer control to the specialized agent."""
        # Using Command(goto=...) communicates to LangGraph to switch nodes
        # We use graph=Command.PARENT because we are inside a sub-agent node
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "last_active_agent": agent_name, 
            }
        )
    return handoff_tool

# --- 4. Define Agent State ---
class AgentState(MessagesState):
    # Track which agent is currently responsible
    last_active_agent: str

# --- 5. Create Agents ---

# -- Database Specialist --
db_tools = [check_database_latency, create_handoff_tool("triage_agent")]
db_agent = create_react_agent(
    model, 
    db_tools, 
    prompt="You are a Database Specialist. Helper with SQL, latency, and consistency issues. If you can't handle it, transfer back to triage."
)

# -- Network Specialist --
network_tools = [check_network_packet_loss, create_handoff_tool("triage_agent")]
network_agent = create_react_agent(
    model, 
    network_tools, 
    prompt="You are a Network Specialist. Help with connectivity, packet loss, and firewalls. If you can't handle it, transfer back to triage."
)

# -- Triage (Router) --
# Triage doesn't need domain tools, just handoff tools
triage_tools = [
    create_handoff_tool("database_specialist"),
    create_handoff_tool("network_specialist")
]
triage_agent = create_react_agent(
    model, 
    triage_tools, 
    prompt="You are a Support Triage Agent. Analyze the user's issue and transfer to 'database_specialist' or 'network_specialist'. If unsure, ask clarifying questions."
)

# --- 6. Build Graph ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("triage_agent", triage_agent)
workflow.add_node("database_specialist", db_agent)
workflow.add_node("network_specialist", network_agent)

# Define Entry Point based on state
def route_initial_request(state: AgentState) -> Literal["triage_agent", "database_specialist", "network_specialist"]:
    # Check if we have an active agent from previous turn
    active = state.get("last_active_agent")
    if active:
        return active
    return "triage_agent"

workflow.add_conditional_edges(START, route_initial_request)

# Compile
app = workflow.compile()

# --- 7. Run Simulation ---
def run_interactive():
    print("--- Cloud Support Multi-Agent System (Type 'q' to quit) ---")
    
    # Initial State
    config = {"configurable": {"thread_id": "1"}}
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["q", "quit"]:
            break
            
        events = app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values"
        )
        
        for event in events:
            if "messages" in event:
                last_msg = event["messages"][-1]
                # Print only AI messages to keep console clean
                if last_msg.type == "ai":
                    print(f"\n[{event.get('last_active_agent', 'System')}]: {last_msg.content}")

if __name__ == "__main__":
    run_interactive()
