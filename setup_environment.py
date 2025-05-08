import os

def load_environment_variables():
    """Load required environment variables for the application."""
    os.environ["OPENAI_API_KEY"] = os.getenv('TWEET-REFLECTOR-AGENT-OPENAI')
    os.environ["LANGCHAIN_API_KEY"] = os.getenv('TWEET-REFLECTOR-AGENT-LANGCHAIN')
    os.environ["LANGSMITH_PROJECT"] = "tweet_reflector_agent"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "reflector agent" 