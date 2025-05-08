import datetime
from typing import Optional, Callable, Any
from setup_environment import load_environment_variables
load_environment_variables()

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

class ResponderChain:
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo",
        max_tokens: Optional[int] = None,
        temperature: float = 0.0
    ):
        self.llm = ChatOpenAI(model=model_name, max_tokens=max_tokens, temperature=temperature)
        self.json_parser = JsonOutputToolsParser(return_id=True)
        self.pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])
        
        self._init_prompts()
        self._init_chains()
    
    def _init_prompts(self):
        self.base_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
   - What information is missing from your answer?
   - What information is superfluous or could be removed?
3. Provide 1-3 specific search queries to research information and improve your answer.

You MUST provide all three components:
1. A detailed answer (~250 words)
2. A reflection with missing and superfluous information
3. 1-3 search queries for improvement""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Answer the user's question above using the required format."),
        ]).partial(
            time=lambda: datetime.datetime.now().isoformat(),
        )
        
        self.first_responder_prompt = self.base_prompt.partial(
            first_instruction="Provide a detailed ~250 word answer."
        )
        
        self.revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""
        
        self.revisor_prompt = self.base_prompt.partial(
            first_instruction=self.revise_instructions
        )
    
    def _init_chains(self):
        self.first_responder = self.first_responder_prompt | self.llm.bind_tools(
            tools=[AnswerQuestion], tool_choice="AnswerQuestion"
        )
        
        self.revisor = self.revisor_prompt | self.llm.bind_tools(
            tools=[ReviseAnswer], tool_choice="ReviseAnswer"
        )
    
    def get_first_responder(self):
        return self.first_responder
    
    def get_revisor(self):
        return self.revisor
    
    def invoke_with_parser(self, message: str, parser: Optional[Callable] = None):
        """Invoke the chain with a specific parser"""
        human_message = HumanMessage(content=message)
        
        if parser is None:
            parser = self.pydantic_parser
            
        chain = self.first_responder_prompt | self.llm.bind_tools(
            tools=[AnswerQuestion], tool_choice="AnswerQuestion"
        ) | parser
        
        return chain.invoke(input={"messages": [human_message]})


if __name__ == "__main__":
    chain = ResponderChain()
    res = chain.invoke_with_parser(
        "Write about AI-Powered SOC / autonomous soc problem domain, "
        "list startups that do that and raised capital."
    )
    print(res)