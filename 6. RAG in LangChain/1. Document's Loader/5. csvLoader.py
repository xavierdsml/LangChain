from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path=r"6. RAG in LangChain\1. Document's Loader\Social_Network_Ads.csv")
docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)