from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI()

# schema 
class Review(TypedDict):
  summary : str
  sentiment: str

structure_model = model.with_structured_output(Review)

result = structure_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdatedcompared to other brands. Hoping for a software update to fix this."""
)

print(result)