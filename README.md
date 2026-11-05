# Hybrid PDF Chatbot

A Streamlit chatbot that uses Groq's `openai/gpt-oss-120b` for answers and summaries. When a PDF is uploaded, it tries the document context first. If the answer is not in the PDF, or no PDF is uploaded, it searches the web with Tavily and asks Groq to answer from the results. Google embeddings and FAISS are used only to index/search PDF text.

## Setup and run in VS Code (Windows)

1. Copy `.env.example` to `.env` and add your API keys:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

   In PowerShell, create the file with `Copy-Item .env.example .env` before editing it.
   `GOOGLE_API_KEY` is only needed when uploading a PDF, for its embeddings. Do not commit `.env`; it is ignored by Git.

2. In the VS Code PowerShell terminal, install and run:

   ```powershell
   Set-Location "C:\Users\lenovo\Desktop\Nlp\chatbot"
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m streamlit run "C:\Users\lenovo\Desktop\Nlp\chatbot\app.py"
   ```

   If Streamlit is already running, stop it with **Ctrl+C** before restarting.
