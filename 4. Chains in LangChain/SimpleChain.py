'Work Flow of the chain :' # prompt -> LLm -> Response

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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
prompt = PromptTemplate(
  template='Generate the five interesting facts about {topic}',
  input_variables=['topic']
)

parser = StrOutputParser()

# chain
chain = prompt | model | parser
result = chain.invoke({'topic':'cricket'})
print(result)

# visulization chain
chain.get_graph().print_ascii()