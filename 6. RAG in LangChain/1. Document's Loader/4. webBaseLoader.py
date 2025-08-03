from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

url = 'https://www.flipkart.com/camlin-kokuyo-student-water-color-tube-5ml-each-12-shades/p/itm7f48316b51885?pid=ARTFSAEPXJBG4R5G&lid=LSTARTFSAEPXJBG4R5GTGQXRK&marketplace=FLIPKART'

loader = WebBaseLoader(url)
docs = loader.load()

# prompt
prompt = PromptTemplate(
  template='Answer the following question in 5 lines \n {question} based on the text given below \n {text}',
  input_variables=['question', 'text']
)

chain = prompt | model | parser
result = chain.invoke({'question':'How many colour are in the box', 'text': docs[0].page_content})

print(result)