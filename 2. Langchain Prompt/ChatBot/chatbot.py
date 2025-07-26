from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="meta-llama/Llama-3.1-8B-Instruct",
  task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# chat history
chat_history = [
  SystemMessage(content='You are a best AI Assistance')
]


while(True):
  user_input = input("You: ")
  chat_history.append(HumanMessage(user_input))

  if user_input == 'exit': break

  result = model.invoke(user_input)
  chat_history.append(AIMessage(result.content))
  print("AI: ",result.content)

print(chat_history)