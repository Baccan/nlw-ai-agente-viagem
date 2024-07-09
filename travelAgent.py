# pip3 install duckduckgo-search
# pip3 install langchain
# pip3 install langchain_openai
# pip3 install wikipedia

# os.environ["OPENAI_API_KEY"] = "sk-...."
import os

# langchain_openai is a module that contains classes to interact with OpenAI's GPT-3 API
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# load_tools is a function that loads tools from the Langchain Hub
from langchain_community.agent_toolkits.load_tools import load_tools

# https://react-lm.github.io/ is a tool to create agents that can react to a given context
from langchain.agents import create_react_agent, AgentExecutor

# hub is a tool to pull pre-trained models from the Langchain Hub
from langchain import hub

# document_loaders is a module that contains classes to load documents from the web
from langchain_community.document_loaders import WebBaseLoader

# vectorstores is a module that contains classes to store and retrieve vectors
from langchain_community.vectorstores import Chroma

# bs4 is a library to parse HTML and XML documents
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableSequence


llm = ChatOpenAI(model="gpt-3.5-turbo")

query = """
Vou viajar para Londres em agosto de 2024.
Quero que faça para um roteiro de viagem para mim com eventos que irão ocorrer na data da viagem e com o preço de passagem de São Paulo para Londres.
"""


# researchAgent is an agent that searches for information on the web and returns a context that contains the information found by the agent in the web
def researchAgent(query, llm):
    tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, prompt=prompt, verbose=True
    )
    webContext = agent_executor.invoke({"input": query})
    return webContext["output"]


# print(researchAgent(query, llm))


# loadData is a function that loads documents from the web and returns a retriever that can be used to retrieve relevant documents
# RAG is a model that retrieves relevant documents from a corpus of documents based on a given query
def loadData():
    loader = WebBaseLoader(
        web_paths=("https://www.dicasdeviagem.com/inglaterra/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=(
                    "postcontentwrap",
                    "pagetitleloading background-imaged loading-dark",
                )
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever


# getRelevantDocs is a function that retrieves relevant documents from the web based on a given query
def getRelevantDocs(query):
    retriever = loadData()
    relevant_documents = retriever.invoke(query)
    print(relevant_documents)
    return relevant_documents


# supervisorAgent is an agent that receives a query, a context, and relevant documents and returns a response
def supervisorAgent(query, llm, webContext, relevant_documents):
    prompt_template = """
  Você é um gerente de uma agência de viagens. Sua resposta final deverá ser um roteiro de viagem completo e detalhado. 
  Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro.
  Contexto: {webContext}
  Documento relevante: {relevant_documents}
  Usuário: {query}
  Assistente:
  """
    # prompt_template is a template that contains placeholders for the context, relevant documents, and query for the supervisorAgent
    prompt = PromptTemplate(
        input_variables=["webContext", "relevant_documents", "query"],
        template=prompt_template,
    )

    sequence = RunnableSequence(prompt | llm)
    response = sequence.invoke(
        {
            "webContext": webContext,
            "relevant_documents": relevant_documents,
            "query": query,
        }
    )
    return response


#  getResponse is a function that receives a query and a language model and returns a response
def getResponse(query, llm):
    webContext = researchAgent(query, llm)
    relevant_documents = getRelevantDocs(query)
    response = supervisorAgent(query, llm, webContext, relevant_documents)
    return response


print(getResponse(query, llm).content)
