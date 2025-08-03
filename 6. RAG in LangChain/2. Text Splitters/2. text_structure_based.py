from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
LangChain’s agent architecture is one of its most powerful features. Agents are LLM-powered decision-makers that can choose from a set of tools to take appropriate actions. For example, an agent can choose to search the web, run Python code, or query a SQL database depending on the user’s query. This dynamic decision-making process enables applications like autonomous assistants, AI data analysts, or multi-functional chatbots. Tools are external APIs or functionalities that the agent can call upon to enhance its capabilities. These could include math solvers, search engines, code execution environments, and more."""

splitter = RecursiveCharacterTextSplitter(
  chunk_size=100,
  chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)