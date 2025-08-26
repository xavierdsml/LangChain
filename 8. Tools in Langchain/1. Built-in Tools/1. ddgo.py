from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
load_dotenv()

# Duck-DuckGo Search

search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke('bigboss news');
print(results)