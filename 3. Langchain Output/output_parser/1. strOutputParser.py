from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
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

parser = StrOutputParser()

# chain ...PIPELINE
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic':'black hole'})
print(result)