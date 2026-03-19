"""Quick test: OpenAI Responses API with web_search tool."""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{"type": "web_search"}],
    input="""You are a high-energy YouTube finance creator. 
             Search the web for the closing numbers of the Sensex, Shanghai Exchange, and Nasdaq from the last 24 hours. 
             Also find one major AI tech news story. 
             Synthesize these facts into a 2-minute, engaging script.""",
)

print(response.output_text)
