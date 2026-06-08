import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()
llm = ChatGroq(model = "llama-3.3-70b-versatile",temperature =0)

question = input("ask a question:")

@tool
def calculator_tool(expression:str)->str:
        #tool description -> tell llm what to do
    """Use this to evaluate math expressions. Extract only the math expression."""

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

#llm decides the tool
tool_choice = llm.invoke(
    f"""
    question: {question}

    available tools:
    search_tool
    calculator_tool

    which tool should be used?
    reply only tool name:
    search_tool
    or
    calculator_tool
    """
)

tool_name = tool_choice.content.strip()
if tool_name == "calculator_tool":
    tool_result = calculator_tool.invoke(question)
elif tool_name == "search_tool":
    tool_result = search_tool.invoke(question)
answer = llm.invoke(
    f"""
    question:
    {question}

    tool result:
    {tool_result}

    can you answer?

    """
)
print(answer.content)

