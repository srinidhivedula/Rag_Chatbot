# Hybrid Document Chatbot

A Streamlit chatbot that uses Groq's `openai/gpt-oss-120b` for answers and summaries. Uploaded documents are searched first; if the answer is not in the document, or no document is uploaded, the chatbot searches the web with Tavily and asks Groq to answer from the results. Google embeddings and FAISS index searchable content.

The upload flow accepts PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX/XLSM, ODT/ODP/ODS, RTF, CSV/TSV, JSON/JSONL, HTML/XML, common text and code files, JPG/JPEG/PNG/WEBP/GIF/BMP/TIFF images, and MP4/M4V/MOV/AVI/MKV/WEBM/MPEG/WMV/3GP videos. Images and sampled video frames are described by Google's Gemini vision model; video audio tracks are not transcribed. Legacy Office and OpenDocument files require LibreOffice to be installed. Other file formats can be selected, but the app reports when it has no content extractor for a format.

## Setup and run in VS Code (Windows)

1. Copy `.env.example` to `.env` and add your API keys:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

   In PowerShell, create the file with `Copy-Item .env.example .env` before editing it.
   `GOOGLE_API_KEY` is needed to index uploaded files and to analyze images or video frames. Do not commit `.env`; it is ignored by Git.

2. In the VS Code PowerShell terminal, install and run:

   ```powershell
   Set-Location "C:\Users\lenovo\Desktop\Nlp\chatbot"
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m streamlit run "C:\Users\lenovo\Desktop\Nlp\chatbot\app.py"
   ```

   If Streamlit is already running, stop it with **Ctrl+C** before restarting.
