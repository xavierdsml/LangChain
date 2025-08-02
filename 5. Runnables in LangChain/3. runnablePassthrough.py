from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt1 = PromptTemplate(
  template='Generate the joke on the following topic {topic}',
  input_variables=['topic']
)

prompt2 = PromptTemplate(
  template='Explain the following joke {text}',
  input_variables=['text']
)

joke_generation = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
  'joke' : RunnablePassthrough(),
  'explaination' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_generation, parallel_chain)
result = final_chain.invoke({'topic':'AI'})
print(result)

# see the work flow of the chain
final_chain.get_graph().print_ascii()