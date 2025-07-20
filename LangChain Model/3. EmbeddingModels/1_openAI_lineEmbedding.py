from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model='', dimensions=32)
result = embedding.embed_query("THis is delhi , capital of India")
print(result)