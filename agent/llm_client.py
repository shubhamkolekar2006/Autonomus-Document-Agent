import os
import time
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMUnavailableError(Exception):
    """
    Custom exception raised when the LLM service is unavailable after retries.
    """
    pass

def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """
    Calls the Groq LLM ('llama-3.3-70b-versatile') with system and user prompts.
    
    Features:
    - Loads GROQ_API_KEY using python-dotenv.
    - If GROQ_API_KEY is not configured or is the default placeholder, falls back to a 
      mocked response mode to allow demo runs to complete and output real files.
    - Supports json_mode=True (sets response_format to json_object and instructs model).
    - Implements up to 3 attempts with exponential backoff (1s, 2s, 4s).
    - Raises LLMUnavailableError on complete failure.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    # Check if key is missing or is the default placeholder
    is_mock_mode = not api_key or api_key == "your_key_here"

    if is_mock_mode:
        print("Warning: GROQ_API_KEY is not set or is placeholder. Operating in DEMO/MOCK mode.")
        time.sleep(0.5)  # Simulate API delay
        
        if json_mode:
            # Check if this is a planner call or reflector call
            if "planning" in system_prompt.lower() or "planner" in system_prompt.lower():
                return json.dumps({
                    "document_type": "project plan",
                    "assumptions": [
                        "Assuming a 3-month launch timeline.",
                        "Assuming hybrid mobile app development framework (React Native/Flutter).",
                        "Assuming a dedicated backend engineering team of 3 developers."
                    ],
                    "title": "Mobile Banking App Launch Plan",
                    "sections": [
                        {"heading": "1. Executive Summary", "purpose": "High-level overview of the 3-month launch strategy."},
                        {"heading": "2. Phase 1: Requirements & UX Design", "purpose": "Key deliverables for Weeks 1 to 4."},
                        {"heading": "3. Phase 2: Core Development & Integration", "purpose": "Key milestones for Weeks 5 to 8."},
                        {"heading": "4. Phase 3: QA Testing & App Store Deployment", "purpose": "Milestones and launch checklist for Weeks 9 to 12."}
                    ]
                })
            else:
                # Reflector review call checking specific critique categories
                return json.dumps({
                    "issues_found": [
                        "[1. Executive Summary] Contained off-topic historical details about traditional banking history.",
                        "[4. Phase 3: QA Testing & App Store Deployment] Inconsistent content: contradicts 3-month launch timeline."
                    ],
                    "needs_revision": True
                })
        else:
            # Content generation calls (Executor drafting or Reflector revision rewriting)
            # Return structured dummy content depending on the section heading requested
            if "Executive Summary" in user_prompt:
                if "Revising" in user_prompt or "rewrite" in user_prompt.lower():
                    # Revised version
                    return (
                        "This project plan outlines the high-level roadmap and critical path deliverables for launching "
                        "the mobile banking application within a strict 3-month timeframe. The target launch is "
                        "scheduled for October 24, 2026. The primary objective is to deliver a secure, fast, and user-friendly "
                        "financial app to the market, focusing on core features: account balances, peer-to-peer transfers, "
                        "and biometrics login.\n\n"
                        "By adopting an agile scrum methodology and deploying a cross-functional team of 8 professionals, "
                        "we will streamline development and mitigate key risks around security audits and store approvals. "
                        "Weekly progress checks will ensure milestones are met on time."
                    )
                else:
                    # Initial draft
                    return (
                        "This project plan outlines the high-level roadmap and critical path deliverables for launching "
                        "the mobile banking application within a strict 3-month timeframe."
                    )
            elif "Requirements & UX Design" in user_prompt:
                return (
                    "Phase 1 will focus on user stories definition and wireframing. The team will complete the UI/UX "
                    "designs for iOS and Android platforms, prioritizing intuitive flows for user onboarding and fund transfers.\n\n"
                    "- Complete Figma design systems by Week 3.\n"
                    "- Conduct user validation interviews with a cohort of 15 beta testers.\n"
                    "- Establish secure backend API definitions for integration."
                )
            elif "Development & Integration" in user_prompt:
                return (
                    "Phase 2 kicks off the core sprint cycle. Developers will build the transaction history page, "
                    "security authentication layers, and third-party bank integration systems.\n\n"
                    "All microservices will run on AWS Cloud infrastructure. Continuous Integration (CI) test coverage "
                    "must exceed 85% before migrating builds to staging environments."
                )
            elif "QA Testing & App Store Deployment" in user_prompt:
                if "Revising" in user_prompt or "rewrite" in user_prompt.lower():
                    # Revised version
                    return (
                        "Phase 3 focuses on rigorous testing, security audits, and deployment cycles. The QA team will "
                        "conduct penetration testing and volume stress testing to ensure the app handles high concurrency.\n\n"
                        "Submission to Apple App Store and Google Play Store will begin in Week 11 to allow 7-10 days "
                        "for reviews. Regulatory compliance certificates (PCI-DSS) must be signed off by compliance officers "
                        "prior to production deployment, ensuring complete security for user financial data."
                    )
                else:
                    # Initial draft
                    return (
                        "Phase 3 focuses on testing. App Store deployment will begin in Week 11."
                    )
            else:
                return (
                    "This section details additional operational procedures. All resources are aligned to ensure "
                    "adherence to the project scope and quality gates.\n\n"
                    "Operational reviews will take place weekly to monitor task status and burn-down rates."
                )

    client = Groq(api_key=api_key)

    # Build prompt setup
    actual_system_prompt = system_prompt or ""
    if json_mode:
        json_instruction = (
            "\n\nYou must respond ONLY with a valid JSON object. "
            "Do not wrap the JSON in markdown code blocks or fences (do not use ```json ... ```)."
        )
        actual_system_prompt = (actual_system_prompt + json_instruction).strip()

    messages = []
    if actual_system_prompt:
        messages.append({"role": "system", "content": actual_system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    # Request parameters
    kwargs = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # Retry loop with backoffs: 1s, 2s, 4s
    backoffs = [1, 2, 4]
    for attempt in range(1, 4):
        print(f"Calling LLM: Attempt {attempt} of 3...")
        try:
            chat_completion = client.chat.completions.create(**kwargs)
            content = chat_completion.choices[0].message.content
            print(f"LLM call succeeded on attempt {attempt}.")
            return content
        except Exception as e:
            print(f"LLM call failed on attempt {attempt}: {str(e)}")
            if attempt < 3:
                wait_time = backoffs[attempt - 1]
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise LLMUnavailableError(
                    f"Groq API call failed after 3 attempts. Last error: {str(e)}"
                ) from e

class LLMClient:
    """
    Backward-compatible client class for communicating with the Groq API.
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("Warning: GROQ_API_KEY is not set in environment variables.")

    def complete(self, prompt: str, system_prompt: str = None, model: str = "llama-3.3-70b-versatile") -> str:
        """
        Send a completion request to Groq using the new retry system.
        """
        try:
            return call_llm(system_prompt=system_prompt, user_prompt=prompt)
        except Exception as e:
            return f"Error during LLM completion: {str(e)}"
