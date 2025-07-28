from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
  Name: str = Field(description='Name of the person')
  Age: int = Field(gt = 18, description='Age of the person')
  City: str = Field(description='Name of the city the person belong to')


parser = PydanticOutputParser(pydantic_object=Person)

# Prompt-template
template = PromptTemplate(
  template='Generate the name, age and city of a ficitonal {place} person \n {format_instruction}',
  input_variables=['place'],
  partial_variables={'format_instruction':parser.get_format_instructions()}
)

# without chain
# prompt = template.invoke({'place':'indian'})
# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)

# with chain
chain = template | model | parser
final_result = chain.invoke({'place':'American'})
print(final_result)