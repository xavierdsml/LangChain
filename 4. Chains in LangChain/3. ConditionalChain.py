from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='mistralai/Mistral-7B-Instruct-v0.3',
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)


# structured-output classification
class Feedback(BaseModel):
  Sentiment : Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser1 = PydanticOutputParser(pydantic_object=Feedback)
parser2 = StrOutputParser()

prompt1 = PromptTemplate(
  template='Classify the sentiment of the following feedback text in to positive or negative \n{feedback} \n {format_instruction}',
  input_variables=['feedback'],
  partial_variables={'format_instruction':parser1.get_format_instructions()}
)

# prompt for Review
prompt2 = PromptTemplate(
  template='Write a appropiate response for the positive feedback \n {feedback}',
  input_variables=['feedback']
)

prompt3 = PromptTemplate(
  template='Write a appropiate response for the negative feedback \n {feedback}',
  input_variables=['feedback']
)


# chain
classifer_chain = prompt1 | model | parser1
branch_chain = RunnableBranch(
  (lambda x: x.Sentiment == 'positive', prompt2 | model | parser2),
  (lambda x: x.Sentiment == 'negative', prompt3 | model | parser2),
  RunnableLambda(lambda x: "could not found sentiment") 
)

chain = classifer_chain | branch_chain
result = chain.invoke({'feedback':'This is a best phone'})
print(result)

# work flow
chain.get_graph().print_ascii()