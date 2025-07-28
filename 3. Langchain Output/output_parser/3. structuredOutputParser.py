from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id = 'mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# making schema
schema = [
  ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
  ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
  ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

# prompt
template = PromptTemplate(
  template='Give 3 facts about the {topic} \n {format_instruction}' ,
  input_variables=['topic'],
  partial_variables={'format_instruction':parser.get_format_instructions()}
)

# without chain
# prompt = template.invoke({'topic':'black hole'})
# result = model.invoke(prompt)
# print(result.content) -- cheking the output

# final_result = parser.parse(result.content)
# print(final_result)


# with chain
chain = template | model | parser
result = chain.invoke({'topic':'black hole'})
print(result)