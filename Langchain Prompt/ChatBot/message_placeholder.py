# previous message load Process in any chat-bot
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# STEP -01  chat template
chat_template = ChatPromptTemplate([
  ('system', 'You are a helful customer agent'),
  MessagesPlaceholder(variable_name='chat_history'),
  ('human', '{query}')
])


# STEP -02 load history
chat_history = []
with open(r'C:\Users\Lenovo\Desktop\LangChain\Langchain Prompt\ChatBot\chat_history.txt') as f:
  chat_history.extend(f.readlines())

print(chat_history)


# STEP -03 Create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'where is my refund'})
print(prompt)