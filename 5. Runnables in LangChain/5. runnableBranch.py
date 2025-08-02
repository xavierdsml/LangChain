from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser() 

prompt1 = PromptTemplate(
  template='Generate the detailed report on the following topic {topic}',
  input_variables=['topic']
)

prompt2 = PromptTemplate(
  template='summarise the following text with in 500 words {text}',
  input_variables=['text']
)

# chain
seqChain = RunnableSequence(prompt1, model, parser)
branChain = RunnableBranch(
  (lambda x:len(x.split())>500, RunnableSequence(prompt2, model, parser)),
  RunnablePassthrough()
)

final_chain = RunnableSequence(seqChain, branChain)
result = final_chain.invoke({'topic':'AI'})
print(result)