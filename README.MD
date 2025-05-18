# 🤖 AI Chatbot Application

This is a simple AI chatbot application built using **Streamlit** for the frontend and **FastAPI** for the backend. The application allows users to interact with AI models and receive intelligent responses based on their queries.

---

## ✨ Features

- 🧠 Chat with AI models (OpenAI GPT-4 and Groq LLaMA)
- 💬 User-friendly interface for entering system prompts and queries
- 🌐 Optional web search for enriched answers
- 📋 Structured response display
- ⚡ FastAPI backend for efficient processing

---

## 🛠️ Technologies Used

- **Frontend**: Streamlit  
- **Backend**: FastAPI  
- **AI Models**: OpenAI GPT-4, Groq LLaMA  
- **Languages/Libraries**: Python, `requests`, `pydantic`  

---

## 🚀 Installation

Follow these steps to run the application locally:

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Set Up a Virtual Environment (optional but recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Backend (FastAPI)

```bash
uvicorn main:app --host 127.0.0.1 --port 9999 --reload
```

### 5. Run the Frontend (Streamlit)

```bash
streamlit run app.py
```

---

## 🌐 Usage

1. Open your browser and go to:  
   [http://localhost:8501](http://localhost:8501)

2. Enter:
   - A **system prompt**
   - A **user query**
   - Choose an **AI model** (OpenAI or Groq)

3. Click **Submit**, and view the AI's response below.

---

## 📡 API Endpoint

The backend API is accessible at:

```
http://127.0.0.1:9999/chat
```

### Request Format (JSON)

```json
{
  "system_prompt": "You are a helpful assistant.",
  "user_query": "What is the capital of France?",
  "provider": "openai",
  "web_search": false
}
```

---

## 🖼️ Screenshot

Here is what the chatbot interface looks like:

![Chat Interface](images.png)



---

## 👤 Author

**Muhmmad Shaban**  
📧 mshaban0121@gmail.com  

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.
