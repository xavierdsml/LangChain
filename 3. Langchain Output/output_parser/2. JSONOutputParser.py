from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

# template ..JSONOutputParser
template = PromptTemplate(
  template='Give me the name, age & city of the fictional person \n {format_instruction}',
  input_variables=[],
  partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.format()
result = model.invoke(prompt) # we can also print to check the LLM Response

final_result = parser.parse(result.content)

# we can also implement with chain
#chain = template | model | parser
#result = chain.invoke({})
#print(result)

print(final_result)
print(type(final_result))