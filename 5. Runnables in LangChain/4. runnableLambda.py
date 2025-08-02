from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser() 

prompt = PromptTemplate(
  template='Generate the joke on the following topic in two lines{topic}',
  input_variables=['topic']
)

# function to count the words -> runnaable
def word_count(text):
  return len(text.split())

runnableWordCount = RunnableLambda(word_count)

# chain
sequChain = RunnableSequence(prompt, model, parser)
paraChain = RunnableParallel({
  'joke': RunnablePassthrough(),
  'word_count': runnableWordCount
})

finalChain = RunnableSequence(sequChain, paraChain)
result = finalChain.invoke({'topic':'AI'})
print(result)