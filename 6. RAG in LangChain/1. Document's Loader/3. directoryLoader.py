from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
  path=r"6. RAG in LangChain\1. Document's Loader\PDF",
  glob='*.pdf',
  loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)