# SETTING UP API KEYS
# This file is used to set up the API keys for the AI agent.
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY=os.environ.get("GROQ_API_KEY") 
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY") 
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY") 

# SETTING UP THE TOOLS
from langchain_groq import ChatGroq
# from langchain_tavily import ChatTavily
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
# settting up the tools
openai = ChatOpenAI(model="gpt-4o-mini")
groq = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
                
search_tool = TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY)

# SETTING UP THE AGENT
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage


system_prompt="Act as an AI chatbot who is smart and friendly"

agent=create_react_agent(
    model=groq,
    tools=[search_tool],
    state_modifier=system_prompt
)




# Function to format search results into a readable answer
def format_search_results(results):
    if not results:
        return "No results found."
    
    formatted_results = []
    for result in results:
        title = result.get('title', 'No Title')
        url = result.get('url', '')
        content = result.get('content', 'No Content')
        formatted_results.append(f"**{title}**: {content[:100]}... [Read more]({url})")
    
    return "\n\n".join(formatted_results)

# Directly using the search tool to get results

query = "Tell me about the trnds in crypto market?"

search_results = search_tool.invoke({"query": query})

# Debugging: Print the search results
if search_results:
    # print("Search Result s:", search_results)
    formatted_response = format_search_results(search_results)
    print("Formatted Response:", formatted_response)
else:
    print("No results found.")
