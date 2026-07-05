from agent.llm_client import call_llm

def execute_plan(user_request: str, plan: dict) -> dict:
    """
    Drafts individual sections of the document based on the plan.
    For each section, calls call_llm to write the actual content.
    """
    title = plan.get("title", "Untitled Document")
    doc_type = plan.get("document_type", "business report")
    assumptions = plan.get("assumptions", [])
    sections = plan.get("sections", [])

    assumptions_str = "\n".join([f"- {a}" for a in assumptions]) if assumptions else "None"

    generated_sections = []

    # INTENTIONAL PER ASSIGNMENT RULES: Using plausible mock data (names, dates, numbers, figures)
    # for any details not explicitly provided in the user request to make the draft robust.
    system_prompt = (
        "You are an autonomous document drafting agent.\n"
        "Your task is to write high-quality, professional content for a specific section of a document. "
        "Adhere to a professional business/technical tone. Keep it highly concise. Write only 1-2 brief paragraphs "
        "or a bullet list of maximum 5 key items. Do not expand into long prose or general theory.\n\n"
        "CRITICAL CONCISENESS & ACTIONABILITY RULES:\n"
        "- Stay extremely concise and direct. Do not add conversational intro/outro text.\n"
        "- Follow the user's requested format and parameters exactly.\n"
        "- Avoid unrelated historical background details, introductory fluff, or generic text.\n"
        "- Prioritize actionable content, concrete deliverables, and specific tasks.\n\n"
        "IMPORTANT: When the user request doesn't specify concrete details like names, dates, amounts, "
        "or numbers, you MUST invent plausible, mock placeholders (e.g. 'Project Alpha', 'October 24, 2026', "
        "'$15,000') to make the document feel realistic and complete, rather than leaving empty placeholders."
    )

    for section in sections:
        heading = section.get("heading", "")
        purpose = section.get("purpose", "")

        user_prompt = (
            f"Document Title: {title}\n"
            f"Document Type: {doc_type}\n"
            f"Assumptions:\n{assumptions_str}\n"
            f"Overall Goal: {user_request}\n\n"
            f"--- Current Section to Draft ---\n"
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

    def execute_plan(self, plan: dict) -> dict:
        """
        Execute the document generation plan. Offers backward compatibility with the skeleton call.
        """
        # Fallback if called with old format: {'prompt': ...} instead of (user_request, plan)
        user_request = plan.get("prompt", "Generate document")
        if "sections" not in plan:
            dummy_plan = {
                "title": "Document Draft",
                "document_type": "proposal",
                "sections": [
                    {"heading": "1. Overview", "purpose": "Overview of the project"}
                ]
            }
            return execute_plan(user_request, dummy_plan)
        return execute_plan(user_request, plan)
