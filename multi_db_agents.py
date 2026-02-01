import os
import sqlite3
from typing import Literal, Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import warnings

# --- 0. Suppress Deprecation Warnings (LangGraph 1.0 specific) ---
warnings.filterwarnings("ignore", message=".*create_react_agent has been moved.*")

# Load env for OPENAI_API_KEY
load_dotenv()

# --- 1. Setup Model and Databases ---
model = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# SQLite Setup (Operational Data)
def setup_sqlite():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE inventory (item_name TEXT, quantity INTEGER, region TEXT)")
    cursor.executemany("INSERT INTO inventory VALUES (?, ?, ?)", [
        ("Server-A", 10, "SG"),
        ("Server-B", 5, "US"),
        ("Switch-C", 20, "SG"),
        ("Firewall-X", 2, "UK"),
        ("PatchPanel-Y", 50, "TH"),
        ("Database-Z", 1, "SG")
    ])
    conn.commit()
    return conn

db_conn = setup_sqlite()

# Vector DB Setup (Knowledge Base)
def setup_vector_db():
    texts = [
        "To fix database latency in Singapore, check the read-replica sync status.",
        "Packet loss in US region is often caused by the Tier-1 provider maintenance on Sundays.",
        "Standard operating procedure for SQL slow queries: Check missing indexes first.",
        "If the firewall blocks port 443, traffic will be dropped in the UK region.",
        "Standard procedure for PatchPanel maintenance in Thailand: Notify the duty manager 2 hours prior.",
        "High CPU on Database-Z in Singapore usually indicates a long-running batch job."
    ]
    vector_db = Chroma.from_texts(texts, embeddings, collection_name="knowledge_base")
    return vector_db

vector_db = setup_vector_db()

# --- 2. Define Tools for Specialists ---

# -- SQLite Specialist Tools --
@tool
def query_inventory(item_name: str):
    """Query the inventory database for item quantity and region."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM inventory WHERE item_name=?", (item_name,))
    result = cursor.fetchone()
    if result:
        return f"Inventory: {result[0]} has {result[1]} units in {result[2]}."
    return "Item not found in inventory."

# -- Knowledge Specialist Tools --
@tool
def search_knowledge_base(query: str):
    """Search the technical knowledge base for troubleshooting steps or SOPs."""
    docs = vector_db.similarity_search(query, k=1)
    if docs:
        return f"Knowledge Base Result: {docs[0].page_content}"
    return "No relevant documentation found."

# --- 3. Define Handoff Tools ---

def create_handoff_tool(agent_name: str):
    """Creates a tool that transfers control to another agent."""
    @tool(f"transfer_to_{agent_name}")
    def handoff_tool():
        """Transfer control to the specialized agent."""
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"last_active_agent": agent_name}
        )
    return handoff_tool

# --- 4. Define Agent State ---
class AgentState(MessagesState):
    last_active_agent: str

# --- 5. Create Agents ---

# -- Inventory Specialist (SQLite) --
inventory_tools = [query_inventory, create_handoff_tool("triage_agent")]
inventory_agent = create_react_agent(
    model, 
    inventory_tools, 
    prompt="You are an Inventory Specialist. You have access to a SQL database of server hardware. Use your tools to check stock."
)

# -- Knowledge Specialist (Vector DB) --
knowledge_tools = [search_knowledge_base, create_handoff_tool("triage_agent")]
knowledge_agent = create_react_agent(
    model, 
    knowledge_tools, 
    prompt="You are a Technical Knowledge Specialist. You have access to a Vector Database with SOPs and troubleshooting guides."
)

# -- Triage (Supervisor) --
triage_tools = [
    create_handoff_tool("inventory_specialist"),
    create_handoff_tool("knowledge_specialist")
]
triage_agent = create_react_agent(
    model, 
    triage_tools, 
    prompt="You are a Support Supervisor. Depending on the user's request, transfer to 'inventory_specialist' (for hardware/stock) or 'knowledge_specialist' (for troubleshooting/SOPs)."
)

# --- 6. Build Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("triage_agent", triage_agent)
workflow.add_node("inventory_specialist", inventory_agent)
workflow.add_node("knowledge_specialist", knowledge_agent)

def route_initial_request(state: AgentState):
    return state.get("last_active_agent", "triage_agent")

workflow.add_conditional_edges(START, route_initial_request)

app = workflow.compile()

if __name__ == "__main__":
    # Interactive loop for direct testing
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["q", "quit"]: break
        for chunk in app.stream({"messages": [HumanMessage(content=user_input)]}, config):
            for node, output in chunk.items():
                if "messages" in output:
                    msg = output["messages"][-1]
                    if msg.type == "ai" and msg.content:
                        print(f"\n[{node}]: {msg.content}")
