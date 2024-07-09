# pip3 install duckduckgo-search
# pip3 install langchain
# pip3 install langchain_openai
# pip3 install wikipedia

import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent

llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = load_tools(["ddg-search", "wikipedia"], llm=llm)

# print(tools[0].name, tools[0].description)
# quit()

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

print(" - - - - -")
# print(agent.agent.llm_chain.prompt.template)
# quit()


query = """
Vou viajar para Londres em agosto de 2024.
Quero que faça para um roteiro de viagem para mim com eventos que irão ocorrer na data da viagem e com o preço de passagem de São Paulo para Londres.
"""


agent.run(query)
