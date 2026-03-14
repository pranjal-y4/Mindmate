from typing import Dict, Any, Tuple

def evaluate_guardrails(tox_result: Dict[str, Any], crisis_result: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Evaluates toxicity and crisis results against strict guardrails logic.
    Returns:
        (guardrails_activated: bool, escalation_message: str)
        If guardrails_activated is False, the message is empty and pipeline proceeds.
    """
    tox_bucket = tox_result.get("severity_bucket", "low")
    crisis_bucket = crisis_result.get("severity_bucket", "low")

    # Hard Guardrail (Immediate Activation)
    if tox_bucket == "high" and crisis_bucket == "high":
        msg = ("I’m really sorry you’re going through this. Your safety matters most right now. "
               "Please contact your local emergency services or mental health crisis line immediately. "
               "If possible, reach out to a trusted friend, family member, doctor, or local health service department right now.")
        return True, msg

    # Escalation Guardrail
    if crisis_bucket == "high" and tox_bucket != "high":
        msg = ("It sounds like things are incredibly heavy right now, and I want to make sure you get the support you need. "
               "Please reach out to a trusted professional or emergency services. "
               "You don't have to carry this alone.")
        return True, msg

    # De-escalation Guardrail
    if tox_bucket == "high" and crisis_bucket != "high":
        msg = ("I understand you are feeling intensely right now, but I must set a boundary against hostile language. "
               "I'm here to provide a safe, calm environment. If you'd like to talk about what's upsetting you "
               "without aggressive language, I am here to listen.")
        return True, msg

    # Pass-Through (Proceed to LLM)
    return False, ""
