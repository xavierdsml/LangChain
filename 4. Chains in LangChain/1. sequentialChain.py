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

# prompt
prompt1 = PromptTemplate(
  template='Give me the detailed report on {topic}',
  input_variables=['topic']
)

prompt2 = PromptTemplate(
  template='Extract the five pointer summary from the report \n {text}',
  input_variables=['text']
)

# parser
parser = StrOutputParser()

# chain
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic':'unemployment in India'})
print(result)

# flow of the chain
chain.get_graph().print_ascii()