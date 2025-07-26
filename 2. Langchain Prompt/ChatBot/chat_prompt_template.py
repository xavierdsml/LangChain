from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="meta-llama/Llama-3.1-8B-Instruct",
  task="task-generation"
)
model = ChatHuggingFace(llm=llm)



# chat-prompt template
chat_template = ChatPromptTemplate([
  ('system', 'You are a helpful {domain} expert'),
  ('human', 'Explain in sinple terms, what is {topic}')

])

prompt= chat_template.invoke({'domain':'cricket'}, {'topic':'Dusara'})
print(prompt)