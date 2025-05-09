# research_graph.py
from typing import List, Optional, Callable

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

class ResearchGraphBuilder:
    def __init__(self, 
                 first_responder,
                 tool_executor,
                 revisor,
                 max_iterations: int = 2):
        self.first_responder = first_responder
        self.tool_executor = tool_executor
        self.revisor = revisor
        self.max_iterations = max_iterations
        self.graph = None
    
    def build(self) -> MessageGraph:
        """Build and return the research graph."""
        builder = MessageGraph()
        
        # Add nodes
        builder.add_node("draft", self.first_responder)
        builder.add_node("execute_tools", self.tool_executor)
        builder.add_node("revise", self.revisor)
        
        # Add standard edges
        builder.add_edge("draft", "execute_tools")
        builder.add_edge("execute_tools", "revise")
        
        # Define and add conditional edge
        builder.add_conditional_edges("revise", self._event_loop)
        
        # Set entry point
        builder.set_entry_point("draft")
        
        # Compile and return
        self.graph = builder.compile()
        return self.graph
    
    def _event_loop(self, state: List[BaseMessage]) -> str:
        """Determine whether to continue the loop or end."""
        count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
        if count_tool_visits > self.max_iterations:
            return END
        return "execute_tools"
    
    def get_mermaid_diagram(self) -> str:
        """Return a Mermaid diagram of the graph."""
        if self.graph is None:
            self.build()
        return self.graph.get_graph().draw_mermaid()
    
    def invoke(self, question: str):
        """Run the graph with the given question."""
        if self.graph is None:
            self.build()
        return self.graph.invoke(question)