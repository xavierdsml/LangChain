from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader(r"6. RAG in LangChain\2. Text Splitters\AI-LLM.pdf")
docs = loader.lazy_load()

splitter = CharacterTextSplitter(
  chunk_size=100,
  chunk_overlap=0,
  separator=''
)

result = splitter.split_documents(docs)
print(result[0].page_content)