import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# For debugging: print API keys availability
print("GROQ_API_KEY:", "Loaded" if GROQ_API_KEY else "Not loaded")
print("TAVILY_API_KEY:", "Loaded" if TAVILY_API_KEY else "Not loaded")
print("OPENAI_API_KEY:", "Loaded" if OPENAI_API_KEY else "Not loaded")

# Step 2: Setup LLM & Tools
try:
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages.ai import AIMessage
    
    # Initialize LLM instances with error handling
    try:
        openai_llm = ChatOpenAI(model="gpt-4o-mini")
        print("OpenAI LLM initialized successfully")
    except Exception as e:
        print(f"Error initializing OpenAI LLM: {e}")
        openai_llm = None
        
    try:
        groq_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
        print("Groq LLM initialized successfully")
    except Exception as e:
        print(f"Error initializing Groq LLM: {e}")
        groq_llm = None
        
    # Initialize Search tool with API key
    try:
        search_tool = TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY)
        print("Search tool initialized successfully")
    except Exception as e:
        print(f"Error initializing search tool: {e}")
        search_tool = None
        
except ImportError as e:
    print(f"Import error: {e}")
    # Set defaults to None so the rest of the code can handle missing dependencies
    openai_llm = None
    groq_llm = None
    search_tool = None

system_prompt_default = "Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt=None, provider="Groq"):
    """
    Get a response from an AI agent.
    
    Args:
        llm_id (str): The model ID to use.
        query (str): The user's query.
        allow_search (bool): Whether to allow web search.
        system_prompt (str, optional): The system prompt to use. Defaults to None.
        provider (str, optional): The provider to use. Defaults to "Groq".
        
    Returns:
        str: The AI agent's response.
    """
    print("=== AI AGENT FUNCTION STARTED ===")
    print(f"Parameters: llm_id={llm_id}, query={query}, allow_search={allow_search}, system_prompt={system_prompt}, provider={provider}")
    
    # Initialize response variables
    response_text = None
    
    try:
        # Set default system prompt if none provided
        if system_prompt is None or not system_prompt.strip():
            system_prompt = system_prompt_default
            print(f"Using default system prompt: {system_prompt}")
        else:
            print(f"Using provided system prompt: {system_prompt}")

        # Select LLM based on provider
        if provider == "Groq":
            try:
                llm = ChatGroq(model=llm_id)
                print(f"Using Groq LLM with model: {llm_id}")
            except Exception as e:
                print(f"Error initializing Groq LLM: {e}")
                return f"Error: Could not initialize Groq LLM with model {llm_id}. {str(e)}"
        elif provider == "OpenAI":
            try:
                llm = ChatOpenAI(model=llm_id)
                print(f"Using OpenAI LLM with model: {llm_id}")
            except Exception as e:
                print(f"Error initializing OpenAI LLM: {e}")
                return f"Error: Could not initialize OpenAI LLM with model {llm_id}. {str(e)}"
        else:
            error_msg = f"Provider '{provider}' is not supported."
            print(error_msg)
            return error_msg

        # Set up tools
        tools = []
        if allow_search:
            if search_tool:
                tools = [search_tool]
                print("Search tool added")
            else:
                print("Search tool requested but not available")
        
        # Create agent with error handling
        try:
            agent = create_react_agent(
                model=llm,
                tools=tools,
                state_modifier=system_prompt
            )
            print("Agent created successfully")
        except Exception as e:
            print(f"Error creating agent: {e}")
            return f"Error: Could not create agent. {str(e)}"

        # Prepare state for agent
        state = {"messages": [{"role": "user", "content": query}]}
        print(f"Prepared state for agent: {state}")
        
        # Invoke agent with error handling
        try:
            print("Invoking agent...")
            response = agent.invoke(state)
            print("Agent invoked successfully")
            
            # Debug: Print raw response
            print(f"Raw response type: {type(response)}")
            if isinstance(response, dict):
                print(f"Response keys: {list(response.keys())}")
        except Exception as e:
            print(f"Error invoking agent: {e}")
            return f"Error: The AI agent encountered an error. {str(e)}"

        # Extract AI messages with extensive error handling
        try:
            ai_messages = []
            
            # Check if response is a dictionary and has 'messages' key
            if isinstance(response, dict) and "messages" in response:
                messages = response["messages"]
                
                # Debug: Print messages information
                print(f"Messages type: {type(messages)}")
                if hasattr(messages, '__iter__'):
                    print(f"Messages length: {len(messages) if messages is not None else 'None'}")
                else:
                    print("Messages is not iterable")
                
                # Handle different message types with defensive programming
                if messages is not None and hasattr(messages, '__iter__'):
                    for message in messages:
                        # Debug: Print message information
                        print(f"Processing message: {type(message)}")
                        
                        # Handle dictionary message
                        if isinstance(message, dict):
                            if message.get("role") == "assistant":
                                content = message.get("content", "")
                                if content:
                                    ai_messages.append(content)
                                    print(f"Added dictionary message: {content[:50]}...")
                        
                        # Handle AIMessage object
                        elif hasattr(message, '__class__') and message.__class__.__name__ == 'AIMessage':
                            if hasattr(message, 'content'):
                                content = message.content
                                ai_messages.append(content)
                                print(f"Added AIMessage: {content[:50]}...")
                        
                        # Default handling for other message types
                        elif hasattr(message, '__str__'):
                            content = str(message)
                            ai_messages.append(content)
                            print(f"Added string-converted message: {content[:50]}...")
                else:
                    print("Messages is None or not iterable")
            else:
                print("Response is not a dictionary or doesn't have 'messages' key")
            
            # Generate final response
            if ai_messages:
                response_text = ai_messages[-1]  # Return the last AI message
                print(f"Final response (from ai_messages): {response_text[:100]}...")
            else:
                response_text = "No AI response could be extracted from the agent's reply."
                print("No AI messages found")
        
        except Exception as e:
            print(f"Error extracting AI messages: {e}")
            import traceback
            traceback.print_exc()
            response_text = f"Error processing AI response: {str(e)}"
        
    except Exception as e:
        print(f"Unexpected error in get_response_from_ai_agent: {e}")
        import traceback
        traceback.print_exc()
        response_text = f"An unexpected error occurred: {str(e)}"
    
    print("=== AI AGENT FUNCTION COMPLETED ===")
    return response_text or "No response could be generated."