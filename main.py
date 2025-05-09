# main.py
from chain import ResponderChain
from tool_executor import execute_tools
from research_graph import ResearchGraphBuilder

def main():
    # Create chain instances
    chain = ResponderChain()
    first_responder = chain.get_first_responder()
    revisor = chain.get_revisor()
    
    # Create and configure the graph
    builder = ResearchGraphBuilder(
        first_responder=first_responder,
        tool_executor=execute_tools,
        revisor=revisor,
        max_iterations=2
    )
    
    # Build the graph
    graph = builder.build()
    
    # Optional: Print the diagram
    print(builder.get_mermaid_diagram())
    
    # Run the graph
    question = "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
    res = builder.invoke(question)
    
    # Extract and print the result
    print(res[-1].tool_calls[0]["args"]["answer"])

if __name__ == "__main__":
    main()