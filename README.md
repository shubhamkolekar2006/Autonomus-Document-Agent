# Autonomous Doc Agent

An autonomous multi-agent document generation system built with FastAPI, Groq (LLM), and `python-docx`.

## 🏗️ Multi-Agent Architecture

The project orchestrates a four-stage autonomous pipeline to create, critique, and compile structured Word documents:

```
[User Prompt]
      │
      ▼
┌───────────┐
│  Planner  │ ──► Analyzes requirements, outlines 4-8 sections, and states assumptions.
└─────┬─────┘
      │
      ▼
┌───────────┐
│ Executor  │ ──► Sells out section-by-section drafting using professional tone.
└─────┬─────┘
      │
      ▼
┌───────────┐
│ Reflector │ ──► Performs critical self-review, flags issues, and rewrites flawed sections.
└─────┬─────┘
      │
      ▼
┌───────────┐
│DocGen     │ ──► Compiles the final document with covers, custom margins, and footer numbering.
└───────────┘
      │
      ▼
[Finished .docx]
```

1.  **Planner** ([agent/planner.py](agent/planner.py)): Interprets requests, handles ambiguity by stating explicit assumptions, determines the appropriate document type, and plans the outline.
2.  **Executor** ([agent/executor.py](agent/executor.py)): Drafts content for each planned section in a professional business tone, using plausible mock data (names, dates, numbers) if details are not explicitly provided.
3.  **Reflector** ([agent/reflector.py](agent/reflector.py)): Reviews the full draft, returns a list of issues found, and selectively triggers target rewrites on flagged sections before final packaging.
4.  **DocGenerator** ([agent/doc_generator.py](agent/doc_generator.py)): Takes the sections JSON and formats them into a polished Word file with Calibri font settings, 1-inch margins, cover page layouts, bulleted item splits, and dynamic footer page numbering (first page omitted).

---

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
Create a virtual environment and install project packages:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (cmd):
.\venv\Scripts\activate.bat
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the root folder (or copy from `.env.example`):
```bash
copy .env.example .env
```
Open `.env` and set your Groq API key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```
*Note: If the key is missing or set to the default placeholder, the system runs in a fully functional demo mock-mode to showcase the multi-agent pipeline and generate files.*

### 4. Start the Application
Run the development server using Uvicorn:
```bash
uvicorn main:app --reload
```

The application will be live at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## 🧪 API Endpoints

*   `GET /health` — Simple health check returning `{"status": "ok"}`.
*   `POST /agent` — Orchestrated document generation.
    *   **Body**: `{"request": "Create a project plan for launching a mobile banking app in 3 months"}`
    *   **Response**: Returns document type, assumptions, plan outline, review notes, and the download URL.
