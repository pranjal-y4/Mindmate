from typing import Dict, Any

# Import architectural components
from src.privacy import privacy_layer
from src.safety_models import detoxifier, distilbert
from src.guardrails import evaluate_guardrails
from src.vector_store import cbd_store
from src.llm_engine import middle_llm

def handle_request(raw_user_message: str) -> Dict[str, Any]:
    """
    Main inference pipeline for MindMate.
    Executes the ordered flow defined in the architectural blueprint.
    """
    print(f"\n--- New Request ---")
    print(f"Raw Input: '{raw_user_message}'")
    
    # 1. Privacy Layer PII Removal
    sanitized_message = privacy_layer.sanitize(raw_user_message)
    print(f"Sanitized: '{sanitized_message}'")
    
    # 2. Safety Evaluation (Runs Detoxifier and DistilBERT on sanitized text)
    tox_result = detoxifier.predict(sanitized_message)
    crisis_result = distilbert.predict(sanitized_message)
    
    print(f"Toxicity: {tox_result['severity_bucket'].upper()} "
          f"(Score: {tox_result['score']})")
    print(f"Crisis:   {crisis_result['severity_bucket'].upper()} "
          f"(Score: {crisis_result['score']})")
    
    # 3. Guardrails Evaluation
    guardrails_activated, escalation_msg = evaluate_guardrails(tox_result, crisis_result)
    
    if guardrails_activated:
        print("[!] Guardrails Activated. Halting LLM generation.")
        return {
            "status": "escalated",
            "response": escalation_msg,
            "metadata": {
                "toxicity": tox_result,
                "crisis": crisis_result
            }
        }
        
    # 4. CBD Retrieval (Safe Path)
    print("[+] Input is Safe. Retrieving CBD techniques.")
    relevant_cbd = cbd_store.retrieve(sanitized_message, top_k=2)
    
    # 5 & 6. Prompt Assembly & Final Generation via Middle LLM
    print("[+] Prompting Middle LLM...")
    final_response = middle_llm.generate(
        sanitized_input=sanitized_message,
        cbd_techniques=relevant_cbd,
        toxicity_level=tox_result['severity_bucket'],
        crisis_level=crisis_result['severity_bucket']
    )
    
    return {
        "status": "success",
        "response": final_response,
        "metadata": {
            "toxicity": tox_result,
            "crisis": crisis_result,
            "cbd_used": [t['name'] for t in relevant_cbd]
        }
    }

if __name__ == "__main__":
    # Test cases representing low, medium, and high risk scenarios
    
    print("Initializing MindMate System...")
    
    test_inputs = [
        # Normal query with PII
        "Hi, my name is Alice from London. I've been feeling a bit stressed about work lately.",
        
        # High Toxicity
        "I hate everyone, you're all so stupid and I want to smash something.",
        
        # High Crisis
        "Everything is hopeless. I just want to end it all tonight.",
        
        # High Toxicity and High Crisis
        "I'm going to kill myself and everyone around me is an idiot."
    ]
    
    for t_input in test_inputs:
        result = handle_request(t_input)
        print(f"MindMate Response:\n{result['response']}")
        print("-------------------\n")
