from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model='gpt-4')
result = model.invoke("WHat is the capital of India ?")
print(result) #many more meta-deta
print(result.content) #only result