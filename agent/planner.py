import json
from agent.llm_client import call_llm

def generate_plan(user_request: str) -> dict:
    """
    Generate a document generation plan based on the user's prompt.
    Uses call_llm in JSON mode to return a structured planning dictionary.
    Retries once on JSON parse or API failure.
    """
    system_prompt = (
        "You are an autonomous planning agent for a document-generation task.\n"
        "Your task is to analyze the user request and generate a document plan. "
        "You must interpret the request, even if vague, ambiguous, or missing details, "
        "and explicitly list any assumptions made due to this lack of detail.\n\n"
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

    try:
        response_text = call_llm(system_prompt=system_prompt, user_prompt=user_request, json_mode=True)
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
            response_text = call_llm(system_prompt=stricter_system_prompt, user_prompt=user_request, json_mode=True)
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

    def generate_plan(self, prompt: str) -> dict:
        """
        Generate a document generation plan using the module-level function.
        """
        return generate_plan(prompt)
