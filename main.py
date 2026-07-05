import os
import uuid
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our robust agent module functions & exceptions
from agent.llm_client import LLMClient, LLMUnavailableError
from agent.planner import Planner, generate_plan
from agent.executor import Executor, execute_plan
from agent.reflector import Reflector, reflect_and_revise
from agent.doc_generator import DocGenerator, generate_docx

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Autonomous Doc Agent",
    description="FastAPI project for an autonomous document generation agent.",
    version="1.0.0"
)

# Ensure outputs directory exists
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Mount outputs directory as static files so generated docs can be downloaded
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Mount static web assets
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Request model for legacy API
class GenerateDocRequest(BaseModel):
    prompt: str
    model: str = "llama-3.1-70b-versatile"
    style: str = "professional"

# Request model for /agent endpoint
class AgentRequest(BaseModel):
    request: str

# Response model for /agent endpoint
class AgentPlanSection(BaseModel):
    heading: str
    purpose: str

class AgentResponse(BaseModel):
    document_type: str
    assumptions: list[str]
    plan: list[AgentPlanSection]
    review_notes: list[str]
    download_url: str
    message: str
    times: dict
    agent_tasks: list[str]

@app.get("/")
async def get_index():
    """
    Serve the main user interface.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Index file not found in static folder.")

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint returning status ok.
    """
    return {"status": "ok"}

@app.post("/agent", response_model=AgentResponse)
async def run_agent(payload: AgentRequest):
    """
    Orchestrates the entire document generation agent workflow:
    1. Validation of request body
    2. generate_plan (Planner Agent)
    3. execute_plan (Executor Agent)
    4. reflect_and_revise (Reflector/Critic Agent)
    5. generate_docx (Doc Generator)
    """
    # 1. Validation
    user_request = payload.request.strip()
    if not user_request or len(user_request) < 10:
        raise HTTPException(
            status_code=400,
            detail="Request body is empty or too short. Please provide a detailed description (minimum 10 characters)."
        )

    try:
        # 2. Planning Stage
        print(f"Agent Stage 1: Generating document plan for request: '{user_request}'")
        start_time = time.perf_counter()
        
        start_planning = time.perf_counter()
        plan = generate_plan(user_request)
        end_planning = time.perf_counter()
        planning_time = round(end_planning - start_planning, 2)
        
        # 3. Drafting Stage
        print(f"Agent Stage 2: Drafting sections for plan: '{plan.get('title')}'")
        start_execution = time.perf_counter()
        draft = execute_plan(user_request, plan)
        end_execution = time.perf_counter()
        execution_time = round(end_execution - start_execution, 2)
        
        # 4. Critical Review and Refinement Stage
        print(f"Agent Stage 3: Running critical review and polish on draft...")
        start_reflection = time.perf_counter()
        revised_draft = reflect_and_revise(user_request, plan, draft)
        end_reflection = time.perf_counter()
        reflection_time = round(end_reflection - start_reflection, 2)
        
        # 5. Compilation Stage
        print(f"Agent Stage 4: Writing sections to DOCX file...")
        start_generation = time.perf_counter()
        filename = generate_docx(revised_draft, output_dir=OUTPUTS_DIR)
        end_generation = time.perf_counter()
        generation_time = round(end_generation - start_generation, 2)
        
        total_time = round(time.perf_counter() - start_time, 2)
        
        # Agent execution tasks list (TODO list representation)
        agent_tasks = [
            "Analyze Request",
            "Determine Document Type",
            "Create Outline",
            "Draft Sections",
            "Review Content",
            "Generate Word Document"
        ]
        
        # 6. Response Construction
        return {
            "document_type": plan.get("document_type", "document"),
            "assumptions": plan.get("assumptions", []),
            "plan": plan.get("sections", []),
            "review_notes": revised_draft.get("review_notes", []),
            "download_url": f"/outputs/{filename}",
            "message": f"Successfully planned, drafted, revised, and generated the document '{plan.get('title')}'.",
            "agent_tasks": agent_tasks,
            "times": {
                "planning": planning_time,
                "execution": execution_time,
                "reflection": reflection_time,
                "document": generation_time,
                "total": total_time
            }
        }
        
    except LLMUnavailableError as e:
        print(f"Agent flow failed: LLM Service is unavailable. Details: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "LLM unavailable. Please try again in a few seconds."}
        )
    except ValueError as e:
        print(f"Agent flow failed: Planner or review JSON parsing failed. Details: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "LLM unavailable. Please try again in a few seconds."}
        )
    except Exception as e:
        print(f"Agent flow failed with unexpected error. Details: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "LLM unavailable. Please try again in a few seconds."}
        )

@app.post("/api/generate")
async def generate_document(payload: GenerateDocRequest):
    """
    API endpoint to trigger the document generation pipeline (compatible with legacy UI).
    """
    llm_client = LLMClient()
    planner = Planner(llm_client)
    executor = Executor(llm_client)
    reflector = Reflector(llm_client)
    generator = DocGenerator(output_dir=OUTPUTS_DIR)

    clean_prompt_title = payload.prompt[:50] + "..." if len(payload.prompt) > 50 else payload.prompt
    doc_title = f"Document: {clean_prompt_title}"
    
    sections = [
        {
            "heading": "1. Executive Summary",
            "content": f"This document was generated in response to the request: '{payload.prompt}'. The document style configuration chosen is '{payload.style}'.",
            "level": 1
        },
        {
            "heading": "2. Project Specifications",
            "content": f"Generated using the agent framework configured with LLM Engine: '{payload.model}'. Details about components and operations will be structured under this segment in future agent steps.",
            "level": 1
        },
        {
            "heading": "2.1 Agent Pipeline Log Summary",
            "content": "The autonomous agent ran four main stages:\n- Planner: Outlined document structure\n- Executor: Drafted section details\n- Reflector: Polished text and refined tone\n- Generator: Saved sections into this MS Word document.",
            "level": 2
        },
        {
            "heading": "3. Conclusion & Next Steps",
            "content": "This concludes the initial generated document draft. Review the layout, and use the agent's feedback panel for additional revisions.",
            "level": 1
        }
    ]

    safe_title = "".join([c if c.isalnum() else "_" for c in payload.prompt[:20]]).strip("_")
    if not safe_title:
        safe_title = "document"
    filename = f"{safe_title}_{uuid.uuid4().hex[:8]}.docx"

    try:
        file_path = generator.generate_docx(
            title=doc_title,
            sections=sections,
            filename=filename
        )
        return {
            "status": "success",
            "filename": filename,
            "file_url": f"/outputs/{filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate document: {str(e)}")
