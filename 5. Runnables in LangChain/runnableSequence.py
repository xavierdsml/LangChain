from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
  template='Write a joke about {topic}',
  input_variables=['topic']
)

prompt2 = PromptTemplate(
  template='Explain the following joke {text}',
  input_variables=['text']
)
parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
print(chain.invoke({'topic':'AI'}))