import os

def load_environment_variables():
    """Load required environment variables for the application."""
    os.environ["OPENAI_API_KEY"] = os.getenv('REFLEXION-AGENT-OPENAI')
    os.environ["LANGCHAIN_API_KEY"] = os.getenv('REFLEXION-AGENT-LANGCHAIN')
    os.environ["LANGSMITH_PROJECT"] = "reflexion_agent"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "reflexion agent" 
    os.environ["TAVILY_API_KEY"] = os.getenv('REFLEXION-AGENT-TAVILY')