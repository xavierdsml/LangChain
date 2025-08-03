from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# pdf loader
loader = PyPDFLoader(r"6. RAG in LangChain\1. Document's Loader\AI-LLM.pdf")
docs = loader.load()

print(docs)
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)