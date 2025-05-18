from pydantic import BaseModel
from typing import Union, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import the fixed agent function
from ai_agent import get_response_from_ai_agent

class RequestState(BaseModel):
    """
    A class to represent the state of a request.
    """
    model_name: str
    model_provider: str
    system_prompt: str
    messages: Union[str, List[str]]  # Allow both str and list
    allow_search: bool

ALLOWED_MODELS = ["gpt-4o-mini", "meta-llama/llama-4-scout-17b-16e-instruct"]

app = FastAPI(title="AI Agent API", description="API for AI Agent", version="1.0")

# Configure CORS to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - replace with actual origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"status": "ok", "message": "API is running"}

@app.post("/chat")
def chat_endpoint(request_state: RequestState):
    """
    Process chat requests and return AI responses.
    
    This endpoint handles:
    - Validating the request
    - Processing the query
    - Calling the AI agent
    - Returning the response in a consistent format
    """
    print("=== Received Request ===")
    print("Model:", request_state.model_name)
    print("Provider:", request_state.model_provider)
    print("System Prompt:", request_state.system_prompt)
    print("Messages Type:", type(request_state.messages))
    print("Allow Search:", request_state.allow_search)

    # Validate model
    if request_state.model_name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Model not allowed. Allowed models: {', '.join(ALLOWED_MODELS)}"
        )

    # Extract parameters
    llm_id = request_state.model_name
    allow_search = request_state.allow_search
    system_prompt = request_state.system_prompt
    provider = request_state.model_provider

    # Process the messages input
    if isinstance(request_state.messages, list):
        try:
            query = " ".join(request_state.messages)
        except Exception as e:
            print(f"Error joining messages list: {e}")
            query = str(request_state.messages)
    else:
        query = request_state.messages or ""

    print("Final Query:", query)

    try:
        # Call AI agent and get response
        response = get_response_from_ai_agent(
            llm_id=llm_id,
            query=query,
            allow_search=allow_search,
            system_prompt=system_prompt,
            provider=provider
        )
        
        # Ensure response is a string
        if response is None:
            response = "The AI agent did not return a response."
        elif not isinstance(response, str):
            response = str(response)
            
        # Return successful response
        return {"response": response}
        
    except Exception as e:
        # Log error details
        import traceback
        print("ERROR IN API ENDPOINT:")
        traceback.print_exc()
        
        # Return error response
        return {
            "error": str(e),
            "response": f"An error occurred while processing your request: {str(e)}"
        }