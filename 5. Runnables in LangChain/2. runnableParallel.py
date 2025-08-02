from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
  template='Generate the tweet for the following topic {topic}',
  input_variables=['topic']
)

prompt2 = PromptTemplate(
  template='Generate the linkedin topic on the following topic {topic}',
  input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
  'tweet' : RunnableSequence(prompt1, model, parser),
  'linkedIn' : RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})
print(result)

# extract 
print(result['tweet'])