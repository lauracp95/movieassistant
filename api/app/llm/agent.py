from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.settings import Settings
from app.llm.prompt import SYSTEM_PROMPT


class MovieAssistant:
    """Simple LLM wrapper for the Movie Night Assistant.
    
    Stateless: each call processes only the current user message.
    """
    
    def __init__(self, settings: Settings) -> None:
        self._llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_deployment,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    
    def chat(self, user_message: str) -> str:
        """Process a user message and return the assistant's reply.
        
        Args:
            user_message: The user's input message.
            
        Returns:
            The assistant's text response.
            
        Raises:
            Exception: If the LLM call fails.
        """
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        
        response = self._llm.invoke(messages)
        return str(response.content)

