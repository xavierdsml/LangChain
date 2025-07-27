from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
  task = 'test-generation'
)

model = ChatHuggingFace(llm=llm)

# 1st prompt --> Detailed report
template1 = PromptTemplate(
  template='Write a detailed report on {topic}',
  input_variables=['topic']
)

# 2nd prompt --> summary
template2 = PromptTemplate(
  template='Give me a five line summary on the following text. {text}',
  input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result1.content})
result2 = model.invoke(prompt2)
print(result2.content)