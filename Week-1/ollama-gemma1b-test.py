from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="gemma3:1b")
response = llm.invoke("Summarize how local models work.")
print(response)