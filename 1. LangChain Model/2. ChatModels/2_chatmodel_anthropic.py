from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

model = ChatAnthropic(mode="");
result = model.invoke("which is the best place in India ? ")
print(result.content)