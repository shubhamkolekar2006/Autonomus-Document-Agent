import json
from agent.llm_client import call_llm

def reflect_and_revise(user_request: str, plan: dict, draft: dict) -> dict:
    """
    Review the full draft and revise sections that contain issues flagged by the reviewer.
    
    1. Sends the full draft to call_llm as a critical reviewer returning:
       { "issues_found": [...], "needs_revision": true/false }
    2. If needs_revision is true, rewrites the flagged sections using call_llm (capped at 1 for speed).
    3. Returns the revised draft with a "review_notes" field.
    """
    # 1. Format full draft for LLM review context
    draft_content_str = ""
    for sec in draft.get("sections", []):
        heading = sec.get("heading", "")
        content = sec.get("content", "")
        draft_content_str += f"\nHeading: {heading}\nContent:\n{content}\n"

    system_prompt = (
        "You are a critical document review agent.\n"
        "Your task is to analyze the full draft of the generated document and determine if it meets the user request.\n\n"
        "Specifically, you must check for and report the following problems:\n"
        "- Missing requested sections (any section planned or explicitly requested that is absent)\n"
        "- Missing assumptions (necessary assumptions that were not documented)\n"
        "- Empty sections (sections with headings but no actual content or body text)\n"
        "- Inconsistent content (contradictory dates, timelines, or specifications between sections)\n"
        "- Off-topic content (unrelated historical background or fluff not requested by the user)\n\n"
        "If any section contains one of these issues, list them in 'issues_found' prefixed with the exact "
        "heading of that section in square brackets, for example: '[1. Introduction] Contained off-topic historical background.'\n\n"
        "CRITICAL FOR SPEED: Be highly selective. Only set needs_revision to true if there is a severe, critical issue. "
        "Otherwise, return needs_revision = false to save processing time.\n\n"
        "You must respond with a JSON object with this exact shape:\n"
        "{\n"
        "  \"issues_found\": [\"[Heading 1] Issue...\", \"[Heading 2] Issue...\"],\n"
        "  \"needs_revision\": true | false\n"
        "}"
    )

    user_prompt = (
        f"User Original Request: {user_request}\n"
        f"Document Type: {plan.get('document_type', 'business report')}\n"
        f"Document Title: {plan.get('title', 'Untitled Document')}\n"
        f"Assumptions: {plan.get('assumptions', [])}\n\n"
        f"--- Full Draft to Review ---\n{draft_content_str}"
    )

    issues_found = []
    needs_revision = False

    try:
        response_text = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
        review_data = json.loads(response_text)
        issues_found = review_data.get("issues_found", [])
        needs_revision = review_data.get("needs_revision", False)
    except Exception as e:
        print(f"Reflector critical review call or parsing failed: {str(e)}. Retrying with stricter instructions...")
        stricter_system_prompt = (
            system_prompt +
            "\n\nCRITICAL: Return ONLY raw valid JSON matching the schema. No markdown wrapping."
        )
        try:
            response_text = call_llm(system_prompt=stricter_system_prompt, user_prompt=user_prompt, json_mode=True)
            review_data = json.loads(response_text)
            issues_found = review_data.get("issues_found", [])
            needs_revision = review_data.get("needs_revision", False)
        except Exception as retry_err:
            print(f"Reflector failed review parsing on retry: {str(retry_err)}. Proceeding without revisions.")
            return {
                "title": draft.get("title", "Untitled"),
                "document_type": draft.get("document_type", "business report"),
                "sections": draft.get("sections", []),
                "review_notes": [f"Review failed to run: {str(retry_err)}"]
            }

    # 2. Iterate and rewrite sections that are flagged (Capped at maximum of 1 revision for speed)
    revised_sections = []
    revised_count = 0
    
    for sec in draft.get("sections", []):
        heading = sec.get("heading", "")
        content = sec.get("content", "")
        
        # Check if the heading is mentioned in issues_found (only rewrite if we haven't hit our speed cap)
        section_issues = []
        if needs_revision and revised_count < 1:
            for issue in issues_found:
                if heading.lower() in issue.lower() or heading.strip().lower() in issue.lower():
                    section_issues.append(issue)
                    
        if section_issues:
            revised_count += 1
            issues_str = "; ".join(section_issues)
            print(f"Reflector: Flagged issues for '{heading}': {issues_str}. Requesting rewrite...")
            
            revision_system_prompt = (
                "You are a professional editor.\n"
                "Your task is to rewrite the provided document section to resolve the issues raised by a reviewer. "
                "Retain a professional tone, and return ONLY the updated body text of the section. "
                "Do not include the heading, introduction, or formatting fences."
            )
            
            revision_user_prompt = (
                f"Original User Request: {user_request}\n"
                f"Section Heading: {heading}\n"
                f"Identified Section Issues: {issues_str}\n\n"
                f"Original Section Content:\n{content}\n\n"
                f"Please rewrite this section to fix the issues, ensuring all details are professional and robust."
            )
            
            try:
                content = call_llm(system_prompt=revision_system_prompt, user_prompt=revision_user_prompt)
            except Exception as e:
                print(f"Reflector failed to revise section '{heading}': {str(e)}. Retaining original content.")
        
        revised_sections.append({
            "heading": heading,
            "content": content
        })

    return {
        "title": draft.get("title", "Untitled"),
        "document_type": draft.get("document_type", "business report"),
        "sections": revised_sections,
        "review_notes": issues_found
    }

class Reflector:
    """
    Reflector Agent: Evaluates drafts, corrects formatting/errors, and suggests refinements.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def reflect(self, content: dict) -> dict:
        """
        Offers backward compatibility with the skeleton call.
        """
        # Fallback if called with old format
        user_request = "Generate document"
        plan = {"title": "Document Draft", "document_type": "proposal"}
        draft = {
            "title": "Document Draft",
            "document_type": "proposal",
            "sections": [{"heading": "1. Overview", "content": content.get("draft", "placeholder")}]
        }
        return reflect_and_revise(user_request, plan, draft)
