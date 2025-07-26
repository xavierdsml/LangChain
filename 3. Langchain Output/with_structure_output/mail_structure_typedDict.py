from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI()

# schema 
class Review(TypedDict):

  Key_themes : Annotated[list[str], "Write down all the key theme discussed in the review in a list"]
  summary : Annotated[str, "A breif summary of the review"]
  sentiment: Annotated[str, "Return sentiment of the review either negative , positive or neutral"]
  pros: Annotated[Optional[list[str]], "Write down all the pros in a list"]
  cons: Annotated[Optional[list[str]], "Write down all the cons in a list"]
  name: Annotated[Optional[str], "Write the name of the reviewer"]


structure_model = model.with_structured_output(Review)

result = structure_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdatedcompared to other brands. Hoping for a software update to fix this."""
)

result2 = structure_model.invoke(""" I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Tushar Gupta""")

print(result)
print(result['summary'])
print(result2['name'])