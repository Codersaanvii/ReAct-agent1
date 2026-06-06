import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
#downloads prompt from langchain hub
from langchain import hub

load_dotenv()

#@ decorator registers the function as a tool so that the agent can call it
@tool
def calculator_tool(expression: str)->str : #returns str
    #tool description -> tell llm what to do
    """Use this to evaluate math expressions. input should be a valid
    python math expression like '1.45 * 2' or '(100+2)/300'."""

    try: 
        #converts str to int for eval
        result = eval(expression)
        return str(result)
    except Exception as e:
        #handles exceptions
        return f"Error: {e}"

@tool
def search_tool(query: str)->str:
    """ Use this to search for current information on the web and when you need to verify the answer or find the answer from the 
    internet before answering.
    When you know the answer,
    DO NOT write Action: None.
    Instead write:
    Final Answer: <answer>"""

    #tavily is the research engine
    from tavily import TavilyClient
    #create search client
    client = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))
    #searches web using query and returns result list
    response = client.search(query = query,max_results = 3)
    results = [r["content"]for r in response["results"]]
    return "\n".join(results)

#agent toolbox
tools = [search_tool, calculator_tool]
#create llm
llm = ChatGroq(model = "llama-3.3-70b-versatile",temperature =0)
#load the react prompt -> this downloads react instructions -> teaches model how to think
prompt = hub.pull("hwchase17/react")
print(prompt)
#create agent
agent = create_react_agent(llm,tools,prompt)

#agent executor runs the loop-> that is a repeated cycle/reasoning loop the agent goes through while solving a question
agent_executor = AgentExecutor(
    agent = agent,
    tools=tools,
    verbose = True,
    max_iterations = 5,
    handle_parsing_error=True
)

while True:
    question = input("ask anything (or 'quit):")
    if question.lower()== 'quit':
        break
    result = agent_executor.invoke({"input":question})
    print(f"answer:{result['output']}")





