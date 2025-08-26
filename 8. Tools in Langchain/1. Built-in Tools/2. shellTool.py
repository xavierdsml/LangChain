from langchain_community.tools import ShellTool

Shell_Tool = ShellTool()
result = Shell_Tool.invoke("whoami")
print(result)