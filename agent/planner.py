import json
from agent.llm_client import call_llm

def generate_plan(user_request: str, source_context: str = "") -> dict:
    """
    Generate a document generation plan based on the user's prompt and optional source context.
    Uses call_llm in JSON mode to return a structured planning dictionary.
    Retries once on JSON parse or API failure.
    """
    system_prompt = (
        "You are an autonomous planning agent for a document-generation task.\n"
        "Your task is to analyze the user request and generate a document plan. "
        "You must interpret the request, even if vague, ambiguous, or missing details, "
        "and explicitly list any assumptions made due to this lack of detail.\n\n"
        "AUTHORITATIVE GROUNDING & SOURCE CONTEXT INSTRUCTIONS:\n"
        "- When Retrieved Source Context is provided, it is the AUTHORITATIVE FACTUAL SOURCE for the document.\n"
        "- Use facts, names, dates, metrics, and technologies present in the retrieved context.\n"
        "- Do NOT invent specific factual details (e.g. company names, executive names, revenues, employee counts, technologies, release dates) not in the source.\n"
        "- Do NOT override source facts using general model knowledge (e.g. if the source states FastAPI, do NOT change it to Django).\n"
        "- Explicitly identify and list in 'assumptions' any important details requested by the user that are NOT provided in the supplied source material (e.g. 'CEO name not provided in source material; left unspecified').\n"
        "- If no source context is provided, formulate reasonable, plausible assumptions as normal.\n\n"
        "CRITICAL CONCISENESS REQUIREMENT: The outline/sections must focus strictly on the user's "
        "requested scope. Prioritize actionable details and avoid planning historical background "
        "or unnecessary fluff sections. If the user asks for a brief roadmap, ensure the planned "
        "sections are direct and concise.\n\n"
        "You must decide the best document TYPE to produce. You MUST choose exactly one of: "
        "proposal, meeting minutes, project plan, business report, technical design, SOP, or product spec.\n\n"
        "Create a highly optimized, concise outline/sections list for the document (strictly between 3 and 4 sections, "
        "logical for the document type). Keeping the number of sections small is critical for generation speed.\n"
        "You must respond with a JSON object with this exact shape:\n"
        "{\n"
        "  \"document_type\": \"proposal | meeting minutes | project plan | business report | technical design | SOP | product spec\",\n"
        "  \"assumptions\": [\"list of assumptions made due to ambiguity/missing info\"],\n"
        "  \"title\": \"string\",\n"
        "  \"sections\": [\n"
        "    {\"heading\": \"string\", \"purpose\": \"one-line description of what this section should contain\"}\n"
        "  ]\n"
        "}"
    )

    user_prompt = user_request
    if source_context and source_context.strip():
        user_prompt = (
            f"User Request: {user_request}\n\n"
            f"--- Retrieved Source Knowledge / Grounding Context ---\n{source_context.strip()}"
        )

    try:
        response_text = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
        return json.loads(response_text)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Planner initial call or JSON load failed: {str(e)}. Retrying with stricter constraints...")
        stricter_system_prompt = (
            system_prompt + 
            "\n\nCRITICAL WARNING: Your response must be parsed directly using python's json.loads(). "
            "Return ONLY the raw JSON object. Do not include markdown codeblocks (```json ... ```), "
            "explanations, or comments."
        )
        try:
            response_text = call_llm(system_prompt=stricter_system_prompt, user_prompt=user_prompt, json_mode=True)
            return json.loads(response_text)
        except Exception as retry_err:
            raise ValueError(
                f"Failed to generate a valid document plan after retry. Detail: {str(retry_err)}"
            ) from retry_err

class Planner:
    """
    Planner Agent: Analyzes the user's document request and plans the structure and content.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def generate_plan(self, prompt: str, source_context: str = "") -> dict:
        """
        Generate a document generation plan using the module-level function.
        """
        return generate_plan(prompt, source_context=source_context)
