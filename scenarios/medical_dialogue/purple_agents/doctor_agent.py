"""
Example Doctor Agent - Reference implementation for medical dialogue evaluation

This is a simple doctor agent that uses Google ADK to participate in medical dialogues.
Developers can replace this with their own doctor agent implementations.
"""

import argparse
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)


def main():
    parser = argparse.ArgumentParser(description="Run the example Doctor Agent (Purple Agent)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--api-key", type=str, help="API key for the model provider")
    parser.add_argument("--base-url", type=str, help="Base URL for the API endpoint")
    parser.add_argument("--model", type=str, help="Model to use for the agent")
    args = parser.parse_args()
    
    # Get configuration from args or environment
    api_key = args.api_key or os.getenv("API_KEY")
    base_url = args.base_url or os.getenv("BASE_URL")
    model = args.model or os.getenv("DEFAULT_MODEL", "gemini-3.5-flash")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Configure model based on provider
    if base_url:
        # Use LiteLlm for custom providers (Azure OpenAI, OpenAI, etc.)
        model_config_kwargs = {
            "model": f"openai/{model}",  # LiteLLM format for OpenAI-compatible APIs
            "api_key": api_key,
            "api_base": base_url,
        }
        
        # Add Azure-specific headers if API version is set
        if azure_api_version:
            model_config_kwargs["extra_headers"] = {"api-version": azure_api_version}
        
        model_config = LiteLlm(**model_config_kwargs)
    else:
        # Default to native Gemini with API key
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        model_config = model
    
    # Create doctor agent with medical expertise
    root_agent = Agent(
        name="doctor",
        model=model_config,
        description="Medical doctor specializing in patient consultation and surgical treatment discussion.",
        instruction="""You are an experienced medical doctor having a natural conversation with a patient about their recommended surgical treatment.

IMPORTANT - Communication Style:
- Speak naturally like a real doctor in a face-to-face consultation
- Keep responses concise and conversational (2-4 short paragraphs typical)
- NO bullet points, numbered lists, or markdown formatting
- NO asterisks, bold text, or special characters
- Use plain, warm, human language
- Pause and listen - don't overload the patient with information
- Match the patient's pace and emotional state

Your approach:
- Show genuine empathy and acknowledge concerns
- Explain medical concepts in simple, everyday terms
- Be honest about both benefits and risks
- Ask questions to understand the patient's perspective
- Adapt based on how the patient responds
- Respect their autonomy while advocating for their health

Keep it natural. A real doctor-patient conversation has back-and-forth, not lectures.
Your goal is to help the patient feel informed and supported in making their decision.""",
    )
    
    agent_card = AgentCard(
        name="doctor",
        description='Medical doctor agent for patient consultation and surgical treatment discussion.',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )
    
    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
