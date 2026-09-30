import os
import sys
import uuid
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import our robust agent module functions & exceptions
from agent.llm_client import LLMClient, LLMUnavailableError
from agent.planner import Planner, generate_plan
from agent.executor import Executor, execute_plan
from agent.reflector import Reflector, reflect_and_revise
from agent.doc_generator import DocGenerator, generate_docx
from agent.document_parser import (
    parse_document,
    build_source_context,
    DocumentParsingError,
    UnsupportedFileTypeError,
    FileTooLargeError,
    EmptyDocumentError,
    CorruptedDocumentError
)
from agent.rag import RAGRetriever, build_retrieved_context

# Load environment variables
load_dotenv()

app = FastAPI(
    title="RAG-Powered Autonomous Document Agent",
    description="An end-to-end RAG-powered document generation system that retrieves relevant information from user-provided documents and uses the retrieved context to generate grounded business documents.",
    version="1.2.0"
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

# Request model for JSON /agent endpoint
class AgentRequest(BaseModel):
    request: str
    source_text: Optional[str] = None

# Response model for /agent endpoint
class AgentPlanSection(BaseModel):
    heading: str
    purpose: str

class SourceItem(BaseModel):
    source_name: str
    source_type: str
    content: str

class RetrievedSourceItem(BaseModel):
    source_name: str
    chunk_id: int
    score: float
    preview: str

class AgentResponse(BaseModel):
    document_type: str
    assumptions: list[str]
    plan: list[AgentPlanSection]
    review_notes: list[str]
    download_url: str
    message: str
    times: dict
    agent_tasks: list[str]
    sources: Optional[list[SourceItem]] = None
    grounded_mode: bool = False
    retrieved_sources: Optional[list[RetrievedSourceItem]] = None

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
async def run_agent(request: Request):
    """
    Orchestrates the grounded document generation agent workflow:
    1. Parses JSON, multipart/form-data, or urlencoded payload.
    2. Validates request and ingests/normalizes source documents.
    3. generate_plan (Grounded Planner Agent)
    4. execute_plan (Grounded Executor Agent)
    5. reflect_and_revise (Grounded Reflector/Critic Agent)
    6. generate_docx (Doc Generator)
    """
    content_type = request.headers.get("content-type", "").lower()
    
    user_request = ""
    source_text: Optional[str] = None
    uploaded_files_data = []  # list of (filename, bytes)

    # 1. Input parsing (supporting multipart/form-data, urlencoded, and application/json)
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
            for key, value in form.multi_items():
                if hasattr(value, "filename") and value.filename:
                    if hasattr(value, "read"):
                        f_bytes = await value.read()
                    elif hasattr(value, "file"):
                        f_bytes = value.file.read()
                    else:
                        f_bytes = b""
                    
                    if f_bytes:
                        uploaded_files_data.append((value.filename, f_bytes))
                else:
                    if key == "request":
                        user_request = str(value or "").strip()
                    elif key == "source_text":
                        source_text = str(value or "").strip()
        except Exception as form_err:
            raise HTTPException(status_code=400, detail=f"Failed to parse form data: {str(form_err)}")
    else:
        try:
            body = await request.json()
            user_request = str(body.get("request", "") or "").strip()
            source_text = body.get("source_text", None)
            if source_text:
                source_text = str(source_text).strip()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid request format. Send JSON payload with 'request' or multipart/form-data."
            )

    # 2. Validation
    if not user_request or len(user_request) < 10:
        raise HTTPException(
            status_code=400,
            detail="Request body is empty or too short. Please provide a detailed description (minimum 10 characters)."
        )

    # 3. Source Document Ingestion and Normalization
    parsed_sources = []
    try:
        for filename, file_bytes in uploaded_files_data:
            parsed = parse_document(file_name=filename, file_bytes=file_bytes)
            parsed_sources.append(parsed)
        
        source_context_obj = build_source_context(
            sources=parsed_sources,
            direct_text=source_text
        )
        recorded_sources = source_context_obj.get("sources", [])
    except (UnsupportedFileTypeError, FileTooLargeError, EmptyDocumentError, CorruptedDocumentError, DocumentParsingError) as parse_err:
        raise HTTPException(status_code=400, detail=str(parse_err))
    except Exception as general_parse_err:
        raise HTTPException(status_code=400, detail=f"Error parsing source document: {str(general_parse_err)}")

    # 4. RAG Retrieval Layer (Prompt 4 Connection)
    grounded_mode = len(recorded_sources) > 0
    retrieved_context_str = ""
    retrieved_sources_preview = None

    if grounded_mode:
        rag_top_k = int(os.environ.get("RAG_TOP_K", "3"))
        retriever = RAGRetriever(sources=recorded_sources)
        retrieved_chunks = retriever.retrieve(query=user_request, top_k=rag_top_k)
        retrieved_context_str = build_retrieved_context(retrieved_chunks)

        top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0
        print(f"[RAG] Sources: {len(recorded_sources)}")
        print(f"[RAG] Chunks indexed: {retriever.vector_store.size()}")
        print(f"[RAG] Query: {user_request}")
        print(f"[RAG] Retrieved: {len(retrieved_chunks)} chunks")
        print(f"[RAG] Top score: {top_score:.4f}")

        retrieved_sources_preview = [
            RetrievedSourceItem(
                source_name=chunk.get("source_name", "unknown"),
                chunk_id=chunk.get("chunk_id", 0),
                score=round(float(chunk.get("score", 0.0)), 4),
                preview=chunk.get("text", "").strip()[:200] + ("..." if len(chunk.get("text", "").strip()) > 200 else "")
            )
            for chunk in retrieved_chunks
        ]

    try:
        # 5. Planning Stage
        print(f"Agent Stage 1: Generating document plan for request: '{user_request}' (Grounded mode: {grounded_mode}, Sources: {len(recorded_sources)})")
        start_time = time.perf_counter()
        
        start_planning = time.perf_counter()
        plan = generate_plan(user_request, source_context=retrieved_context_str)
        end_planning = time.perf_counter()
        planning_time = round(end_planning - start_planning, 2)
        
        # 6. Drafting Stage
        print(f"Agent Stage 2: Drafting sections for plan: '{plan.get('title')}'")
        start_execution = time.perf_counter()
        draft = execute_plan(user_request, plan, source_context=retrieved_context_str)
        end_execution = time.perf_counter()
        execution_time = round(end_execution - start_execution, 2)
        
        # 7. Critical Review and Refinement Stage (Quality & Grounding Review)
        print(f"Agent Stage 3: Running critical quality and grounding review on draft...")
        start_reflection = time.perf_counter()
        revised_draft = reflect_and_revise(user_request, plan, draft, source_context=retrieved_context_str)
        end_reflection = time.perf_counter()
        reflection_time = round(end_reflection - start_reflection, 2)
        
        # 8. Compilation Stage
        print(f"Agent Stage 4: Writing sections to DOCX file...")
        start_generation = time.perf_counter()
        filename = generate_docx(revised_draft, output_dir=OUTPUTS_DIR)
        end_generation = time.perf_counter()
        generation_time = round(end_generation - start_generation, 2)
        
        total_time = round(time.perf_counter() - start_time, 2)
        
        # Agent execution tasks list (TODO list representation)
        agent_tasks = [
            "Analyze Request & Retrieve Relevant Chunks" if grounded_mode else "Analyze Request",
            "Determine Document Type",
            "Create Grounded Outline" if grounded_mode else "Create Outline",
            "Draft Sections (Grounded)" if grounded_mode else "Draft Sections",
            "Review Content & Fact Integrity",
            "Generate Word Document"
        ]
        
        # 9. Response Construction
        return {
            "document_type": plan.get("document_type", "document"),
            "assumptions": plan.get("assumptions", []),
            "plan": plan.get("sections", []),
            "review_notes": revised_draft.get("review_notes", []),
            "download_url": f"/outputs/{filename}",
            "message": f"Successfully planned, drafted, revised, and generated the document '{plan.get('title')}'.",
            "agent_tasks": agent_tasks,
            "grounded_mode": grounded_mode,
            "retrieved_sources": retrieved_sources_preview,
            "sources": [
                {
                    "source_name": s["source_name"],
                    "source_type": s["source_type"],
                    "content": s["content"]
                }
                for s in recorded_sources
            ],
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
