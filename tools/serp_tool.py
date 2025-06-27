from langchain.agents import initialize_agent, Tool
import os
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import AgentExecutor, ZeroShotAgent

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LLM.llm import SarvamChat  

system_message = SystemMessagePromptTemplate.from_template("""
You are an intelligent assistant. You have to think very fast and within 1-2 observations give a fast Fineal Answer. Always follow this format:

- If you want to use a tool, reply ONLY with:
  Thought: <your thought>
  Action: <tool name>
  Action Input: <tool input>

- If you are ready to answer, reply ONLY with:
  Thought: <your thought>
  Final Answer: <your final answer>

DO NOT output both an Action and Final Answer together.
                                                           
                                                           
""")

human_message = HumanMessagePromptTemplate.from_template("{input}")

prompt = ChatPromptTemplate.from_messages([system_message, human_message])

serp_api_key="SERP_API_KEY"
search_tools = SerpAPIWrapper(serpapi_api_key=serp_api_key)

llm = SarvamChat(api_key="SARVAM_API_KEY")
llm_chain = prompt | llm



search = SerpAPIWrapper(serpapi_api_key="SERP_API_KEY")
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for answering questions about current events or the web"
    )
]

agent = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # Optional, if you want automatic retries
)



response = agent_executor.run("Vijay Mallaya kon hein bhai?")
print(response)
