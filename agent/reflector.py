import json
from agent.llm_client import call_llm

def reflect_and_revise(user_request: str, plan: dict, draft: dict, source_context: str = "") -> dict:
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
        "You are a critical document review agent performing dual-stage quality and grounding review.\n"
        "Your task is to analyze the full draft of the generated document and determine if it meets the user request and adheres strictly to the retrieved facts.\n\n"
        "REVIEW CRITERIA:\n"
        "A. Normal Document Quality Review:\n"
        "- Missing requested sections or empty sections.\n"
        "- Inconsistent content (contradictory dates, timelines, or specifications between sections).\n"
        "- Off-topic content (unrelated historical fluff not requested by the user).\n\n"
        "B. Grounding & Factual Integrity Review:\n"
        "- Contradictions: Any claim, technology, date, or number that contradicts the provided Source Context.\n"
        "- Unsupported Claims: Any claim that a company/product currently uses technologies (e.g. Kafka, Redis, Kubernetes) not in the source context.\n"
        "- Invented Missing Information: If the user requested specific details (e.g. CEO name, revenue, employee count) that are NOT in the source context, the draft must NOT invent names or numbers. It must explicitly state that the information was not provided.\n\n"
        "If any section contains one of these issues, list them in 'issues_found' prefixed with the exact "
        "heading of that section in square brackets, for example: '[1. Introduction] Contained invented CEO name not present in source context.'\n\n"
        "CRITICAL FOR SPEED: Be selective. Only set needs_revision to true if there is a severe factual or grounding issue, or missing section. "
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
        f"Assumptions: {plan.get('assumptions', [])}\n"
    )

    if source_context and source_context.strip():
        user_prompt += f"\n--- Retrieved Source Knowledge / Grounding Context ---\n{source_context.strip()}\n"

    user_prompt += f"\n--- Full Draft to Review ---\n{draft_content_str}"

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
                f"Identified Section Issues: {issues_str}\n"
            )

            if source_context and source_context.strip():
                revision_user_prompt += f"\n--- Source Knowledge / Grounding Context ---\n{source_context.strip()}\n"

            revision_user_prompt += (
                f"\nOriginal Section Content:\n{content}\n\n"
                f"Please rewrite this section to fix the issues, ensuring all details are grounded, professional, and robust."
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

    def reflect(self, content: dict, source_context: str = "") -> dict:
        """
        Offers backward compatibility with the skeleton call.
        """
        user_request = "Generate document"
        plan = {"title": "Document Draft", "document_type": "proposal"}
        draft = {
            "title": "Document Draft",
            "document_type": "proposal",
            "sections": [{"heading": "1. Overview", "content": content.get("draft", "placeholder")}]
        }
        return reflect_and_revise(user_request, plan, draft, source_context=source_context)
