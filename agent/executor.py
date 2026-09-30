from agent.llm_client import call_llm

def execute_plan(user_request: str, plan: dict, source_context: str = "") -> dict:
    """
    Drafts individual sections of the document based on the plan and optional source context.
    For each section, calls call_llm to write the actual content.
    """
    title = plan.get("title", "Untitled Document")
    doc_type = plan.get("document_type", "business report")
    assumptions = plan.get("assumptions", [])
    sections = plan.get("sections", [])

    assumptions_str = "\n".join([f"- {a}" for a in assumptions]) if assumptions else "None"

    generated_sections = []

    # System prompt enforcing grounding, missing information handling, and conciseness
    system_prompt = (
        "You are an autonomous document drafting agent.\n"
        "Your task is to write high-quality, professional content for a specific section of a document. "
        "Adhere to a professional business/technical tone. Keep it highly concise. Write only 1-2 brief paragraphs "
        "or a bullet list of maximum 5 key items. Do not expand into long prose or general theory.\n\n"
        "STRICT GROUNDING & SOURCE CONTEXT HIERARCHY:\n"
        "1. Retrieved source information is the AUTHORITATIVE FACTUAL SOURCE for the document.\n"
        "2. Explicit user request.\n"
        "3. Clearly labeled assumptions/recommendations only when necessary.\n\n"
        "CRITICAL FACTUAL INTEGRITY & MISSING INFORMATION RULES:\n"
        "- When Retrieved Source Context is provided, you MUST ground your content strictly in the facts, dates, names, "
        "metrics, and technologies from the source context.\n"
        "- Do NOT modify or contradict explicit facts given in the source context (e.g. if the source says FastAPI, never change to Django).\n"
        "- Do NOT claim that a company or product currently uses technologies (e.g. Redis, Kafka, Kubernetes, AWS ECS) "
        "unless those technologies are explicitly stated in the source context or requested by the user. "
        "You may mention technologies ONLY as optional future recommendations (e.g. 'Redis could be considered for future caching.').\n"
        "- HANDLING MISSING INFORMATION: If the user explicitly asks for specific details (such as CEO name, annual revenue, "
        "employee count, pricing, release date) that are NOT present in the retrieved source context, DO NOT invent names, numbers, or facts. "
        "Instead, explicitly state that the information was not provided (e.g. 'CEO: Not provided in the supplied source material.', "
        "'Annual Revenue: Not provided in the supplied source material.', 'Employee Count: Not provided in the supplied source material.').\n"
        "- If NO source context is provided, you may use reasonable, plausible placeholders.\n\n"
        "CRITICAL CONCISENESS & ACTIONABILITY RULES:\n"
        "- Stay extremely concise and direct. Do not add conversational intro/outro text.\n"
        "- Follow the user's requested format and parameters exactly.\n"
        "- Avoid unrelated historical background details, introductory fluff, or generic text.\n"
        "- Prioritize actionable content, concrete deliverables, and specific tasks."
    )

    for section in sections:
        heading = section.get("heading", "")
        purpose = section.get("purpose", "")

        user_prompt = (
            f"Document Title: {title}\n"
            f"Document Type: {doc_type}\n"
            f"Assumptions:\n{assumptions_str}\n"
            f"Overall Goal: {user_request}\n"
        )

        if source_context and source_context.strip():
            user_prompt += f"\n--- Retrieved Source Knowledge / Grounding Context ---\n{source_context.strip()}\n"

        user_prompt += (
            f"\n--- Current Section to Draft ---\n"
            f"Section Heading: {heading}\n"
            f"Section Purpose: {purpose}\n\n"
            f"Please write the content for this section now. Do not repeat the heading. "
            f"Do not write other sections. Output ONLY the body content for this section."
        )

        print(f"Drafting section: '{heading}'...")
        content = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        
        generated_sections.append({
            "heading": heading,
            "content": content
        })

    return {
        "title": title,
        "document_type": doc_type,
        "sections": generated_sections
    }

class Executor:
    """
    Executor Agent: Drafts individual sections of the document according to the plan.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def execute_plan(self, plan: dict, source_context: str = "") -> dict:
        """
        Execute the document generation plan. Offers backward compatibility with the skeleton call.
        """
        user_request = plan.get("prompt", "Generate document")
        if "sections" not in plan:
            dummy_plan = {
                "title": "Document Draft",
                "document_type": "proposal",
                "sections": [
                    {"heading": "1. Overview", "purpose": "Overview of the project"}
                ]
            }
            return execute_plan(user_request, dummy_plan, source_context=source_context)
        return execute_plan(user_request, plan, source_context=source_context)
