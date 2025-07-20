from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model='', dimensions=32)

documents = [
  'New Delhi is the capital of India',
  'Paris is the capital of France'
]
result = documents.embed_documents(documents)
print(result)