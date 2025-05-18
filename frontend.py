import streamlit as st
import requests

# Page settings
st.set_page_config(
    page_title="Chatbot",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display header
st.title("🤖 Customaizable AI Chatbot")
st.write("Create and interact with your own AI agent.")

# Create two columns - one for chat, one for settings
col1, col2 = st.columns([2, 1])

# Settings column
with col2:
    st.header("Settings")
    
    # Input fields
    system_prompt = st.text_area(
        "System Prompt", 
        placeholder="Enter your system prompt here", 
        value="Act as a knowledgeable assistant.",
        height=100
    )
    
    # Model selection
    st.subheader("Model")
    GROQ_MODELS = ["meta-llama/llama-4-scout-17b-16e-instruct"]
    OPENAI_MODELS = ["gpt-4o-mini"]
    provider = st.radio("Select Model Provider", ["OpenAI", "Groq"])

    if provider == "OpenAI":
        model_name = st.selectbox("Select Model", OPENAI_MODELS)
    elif provider == "Groq":
        model_name = st.selectbox("Select Model", GROQ_MODELS)

    allow_websearch = st.checkbox("Allow Web Search", value=False)
    
    # Status indicator
    st.subheader("Connection Status")
    try:
        response = requests.get("http://127.0.0.1:9999/")
        if response.status_code == 200:
            st.success("Backend connected")
        else:
            st.error("Backend error")
    except:
        st.error("Cannot connect to backend server")

# Chat column
with col1:
    st.header("Chat")
    
    # Display chat messages
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    # Process user input when submitted
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display the latest user message (the one just added)
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # API endpoint
        API_ENDPOINT = "http://127.0.0.1:9999/chat"
        
        # Prepare payload for API request
        payload = {
            "model_name": model_name,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": user_input,
            "allow_search": allow_websearch
        }

        # Show thinking indicator
        with chat_container:
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("_Thinking..._")
                
                try:
                    # Send request to backend
                    response = requests.post(API_ENDPOINT, json=payload, timeout=60)
                    
                    # Process response
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        # Check for error
                        if "error" in response_data and response_data["error"]:
                            assistant_response = f"🚫 Error: {response_data['error']}"
                        # Check for response
                        elif "response" in response_data:
                            assistant_response = response_data["response"]
                        else:
                            assistant_response = "⚠️ No response received from the AI service."
                    else:
                        assistant_response = f"🚫 Error: Server returned status code {response.status_code}"
                        
                except requests.exceptions.RequestException as e:
                    assistant_response = f"🚫 Connection error: {str(e)}"
                except Exception as e:
                    assistant_response = f"🚫 Unexpected error: {str(e)}"
                
                # Update thinking message with actual response
                thinking_placeholder.markdown(assistant_response)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Add a reset button at the bottom
if st.button("Reset Conversation"):
    st.session_state.messages = []
    st.rerun()