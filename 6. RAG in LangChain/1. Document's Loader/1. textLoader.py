from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# text load from the .txt file
loader = TextLoader(r"6. RAG in LangChain\1. Document's Loader\cricket.txt", encoding='utf-8')
docs = loader.load()

# print(docs)
print(docs[0])
print(type(docs[0]))
print(docs[0].metadata, "\n")



# prompt
prompt = PromptTemplate(
  template='Write the summary of the given poem {text}',
  input_variables=['text']
)

chain = prompt | model | parser
result = chain.invoke({'text':docs[0].page_content})
print(result)