import json
import os
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

def _get_hf_token() -> str:
    """Read HF_TOKEN from Streamlit secrets (cloud) or environment variable (local)."""
    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    except Exception:
        return os.environ.get("HF_TOKEN", "")


SYSTEM_BASE = """You are MindMate, a deeply compassionate, warm, and uplifting mental health support companion.

YOUR CORE PERSONALITY:
- You speak like a caring, emotionally intelligent best friend who genuinely wants to help.
- You are endlessly patient, non-judgmental, and always hopeful.
- You NEVER make the user feel weak, broken, or judged for how they feel.
- You speak with warmth, sincerity, and quiet strength.
- You always remind people that their feelings are valid and that things CAN get better.

YOUR TONE RULES:
- Motivating: Remind users of their resilience and inner strength. Use phrases like "You've made it through hard moments before", "That takes real courage to share", "You matter more than you know."
- Empathetic first: Always acknowledge the emotion BEFORE offering any advice or technique.  
- Hopeful: Even in dark moments, hold space for hope. Never be catastrophic or cold.
- Concise but warm: 3–6 sentences for casual, 5–9 for distress. Never robotic, never clinical.
- End invitingly: Always close with a gentle question or encouragement to keep sharing.

WHAT YOU NEVER DO:
- Never diagnose or name a mental illness.
- Never give medical, legal, or medication advice.
- Never make the user feel like a burden.
- Never be dismissive ("I'm sure it'll be fine") — always take their feelings seriously.
- Never force coping techniques into casual greetings.
"""


class MiddleLLMEngine:
    def __init__(self, model_identifier: str = "Qwen/Qwen2.5-72B-Instruct"):
        self.model_identifier = model_identifier
        self.hf_token = _get_hf_token()
        if self.hf_token and InferenceClient:
            self.client = InferenceClient(model=self.model_identifier, token=self.hf_token)
        else:
            self.client = None

    def _is_casual(self, text: str) -> bool:
        casual_phrases = ["hey", "hi", "hello", "sup", "what's up", "howdy",
                          "good morning", "good evening", "good afternoon", "yo"]
        lower = text.lower().strip()
        return any(lower.startswith(p) for p in casual_phrases) and len(lower.split()) < 6

    def build_messages(self, sanitized_input: str, cbd_techniques: list,
                       toxicity_level: str, crisis_level: str,
                       conversation_summary: str = "") -> list:

        cbd_block = "\n".join(
            f"  • {t['name']}: {t['description']}" for t in cbd_techniques
        )

        summary_block = ""
        if conversation_summary and conversation_summary.strip():
            summary_block = f"""
CONVERSATION CONTEXT (what has been shared so far):
{conversation_summary.strip()}
Use this to give personalised, remembered responses — refer back to the user's journey so far.
---
"""

        is_casual = self._is_casual(sanitized_input)
        is_crisis = crisis_level == "high"

        if is_casual:
            style_block = (
                "The user sent a casual greeting. Respond warmly and naturally — ask how they're doing today. "
                "Do NOT mention coping techniques unprompted."
            )
        elif is_crisis:
            style_block = (
                "⚠️ The user may be in significant distress or expressing thoughts of self-harm. "
                "Your response must be deeply compassionate and prioritise their safety.\n"
                "- Acknowledge how much pain they must be in right now.\n"
                "- Remind them their life has value and that they are not alone.\n"
                "- Gently but clearly encourage them to reach out to someone they trust or a professional.\n"
                "- You may mention a crisis line (e.g. Samaritans: 116 123) if appropriate.\n"
                "- Keep the tone soft, loving, and full of hope — never panicked or robotic."
            )
        else:
            style_block = (
                "The user is expressing a feeling or difficulty. Follow this sequence:\n"
                "1. First, VALIDATE their emotion genuinely and warmly — make them feel truly heard.\n"
                "2. Remind them of their courage and strength for sharing.\n"
                "3. Only THEN, weave in ONE of the retrieved coping techniques naturally — as a caring friend would, not a textbook.\n"
                "   Good: 'One thing that sometimes helps me ground myself is...' or 'Some people find that writing things down...'\n"
                "   Bad: 'Technique #1: Grounding:...'\n"
                "4. Close with a warm, open question that invites them to keep sharing."
            )

        system_prompt = f"""{SYSTEM_BASE}
{summary_block}
CURRENT RESPONSE APPROACH:
{style_block}

RETRIEVED COPING TECHNIQUES (use naturally if relevant, max 1 unless asked):
{cbd_block}

SAFETY STATUS: Toxicity={toxicity_level.upper()} | Crisis={crisis_level.upper()}
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": sanitized_input}
        ]

    def generate(self, sanitized_input: str, cbd_techniques: list,
                 toxicity_level: str, crisis_level: str,
                 conversation_summary: str = "") -> str:

        messages = self.build_messages(
            sanitized_input, cbd_techniques, toxicity_level, crisis_level, conversation_summary
        )

        if self.client:
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=400,
                    temperature=0.75,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"HF API Failed: {e}. Falling back to mock.")

        # Smart mock fallback
        lower = sanitized_input.lower().strip()
        greetings = ["hey", "hi", "hello", "sup", "what's up", "howdy"]
        if any(lower.startswith(g) for g in greetings) and len(lower.split()) < 6:
            return "Hey! 👋 So glad you're here. How are you feeling today? I'm all ears and here for you. 💙"

        if crisis_level == "high":
            return (
                "I'm really glad you shared this with me, and I want you to know — what you're feeling is real, "
                "and you don't have to carry it alone. Your life matters deeply. 💙 "
                "Please consider reaching out to someone you trust, or a crisis line like Samaritans (116 123) "
                "who are there 24/7. You deserve support right now. Would you be willing to talk to someone today?"
            )

        technique = cbd_techniques[0] if cbd_techniques else {
            "name": "deep breathing",
            "description": "Try breathing in for 4 counts, hold for 4, exhale for 4. It can really help calm your nervous system."
        }
        return (
            f"Thank you for trusting me with this — sharing takes real courage. 💙 "
            f"What you're feeling makes complete sense, and you're not alone in this. "
            f"Something that sometimes helps when things feel heavy is {technique['name'].lower()} — "
            f"{technique['description']} "
            f"Would you like to talk more about what's been going on? I'm here."
        )

    def summarize_conversation(self, messages: list) -> str:
        if not messages or len(messages) < 2:
            return ""

        convo_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages[-12:]
        )
        summary_messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise summarizer assistant. Given a mental health support conversation, "
                    "write a 2-3 sentence summary capturing: the user's emotional state, key topics shared, "
                    "and any coping approaches discussed. Be neutral, compassionate, and factual."
                )
            },
            {"role": "user", "content": f"Summarise this conversation:\n\n{convo_text}"}
        ]

        if self.client:
            try:
                res = self.client.chat_completion(messages=summary_messages, max_tokens=150, temperature=0.3)
                return res.choices[0].message.content.strip()
            except Exception as e:
                print(f"Summarization failed: {e}")
        return ""
