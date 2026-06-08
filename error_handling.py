import os
import random
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain import hub

load_dotenv()

@tool
def calculator_tool(expression: str) -> str:
    """Use this to evaluate simple math expressions like '1.43 * 2' or '(100 + 200) / 3'."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

failure_count = {"search":0}

@tool
def search_tool(query:str)->str:
    """use this to search for current information on the web"""

    if random.random()<0.3:
        failure_count["search"] +=1
        if failure_count["search"] >=2:
            return "tool failure: search tool has failed twice. Stop trying this tool and tell the user you could not complete the task"
        return "tool failure: search tool currently unavailable, try once more"

    failure_count["search"]=0

    from tavily import TavilyClient

    client = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))
    response = client.search(query = query, max_results = 3)
    results = [r["content"]for r in response["results"]]
    return "\n".join(results)


python_repl = PythonREPLTool()

tools = [search_tool,calculator_tool,python_repl]

llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm,tools,prompt)

agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True,
    max_iterations = 6,
    handle_parsing_errors = True,
    return_intermediate_steps = True
)

benchmark_questions = [
    "What is the population of India multiplied by 9?",
    "What is the GDP of Japan divided by its population?",
    "Write a Python function to calculate fibonacci numbers and run it for n=14",
    "What is the current price of gold per gram in INR?",
    "What is 12% of the population of Mumbai?",
    "Use Python to generate the first 26 prime numbers and sum them",
    "What year was the Eiffel Tower built and how many years ago was that?",
    "What is the square root of the number of countries in the United Nations?",
    "Write and run Python code to calculate compound interest: principal=15382, rate=9%, years=3",
    "What is the distance from Mumbai to Delhi in km divided by the speed of a commercial flight?"
]

def run_benchmark():
    """Run all 10 benchmark tasks and record results"""

    results = []
    print("running benchmark")

    for i,q in enumerate(benchmark_questions,1):
        print(f"{i}/10:{q}")

        try:
            result = agent_executor.invoke({"input":q})
            answer = result["output"]
            steps = len(result.get("intermediate_steps",[]))

            passed = "could not" not in answer.lower() and "unable" not in answer.lower()

            results.append({
                "question":q,
                "answer":answer,
                "steps":steps,
                "passed":passed
            })

            print(f"answer: {answer[:100]}")
            print(f"steps taken: {steps}")

        except Exception as e:
            results.append({
                "question":q,
                "answer":f"crashed: {e}",
                "steps":0,
                "passed":False
            })


    passed = sum(1 for r in results if r["passed"])
    print(f"BENCHMARK RESULTS: {passed}/10 tasks completed")
    

def run_interactive():
    """Interactive mode for manual testing."""
    print("\nReAct Agent ready. Type 'quit' to exit, 'benchmark' to run all 10 tasks.\n")
    while True:
        # Reset failure count each question
        failure_count["search"] = 0

        question = input("Ask anything: ").strip()
        if question.lower() == "quit":
            break
        elif question.lower() == "benchmark":
            run_benchmark()
        else:
            result = agent_executor.invoke({"input": question})
            print(f"\nAnswer: {result['output']}\n")

if __name__ == "__main__":
    run_interactive()

