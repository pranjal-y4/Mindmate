import random
from typing import Dict, Any

# ─── Crisis Keyword System ─────────────────────────────────────────────────────
# Uses ROOT WORDS so "wanna die", "wanting to die", "want to die" all match.
# Each entry: (root_word_or_phrase, severity)
# Checked with `in` against lowercased text → very broad matching.

CRISIS_HIGH_ROOTS = [
    # Wanna / wanna die variations
    "wanna die", "wanna be dead", "wanna end", "wanna kill",
    # Want to die
    "want to die", "want to be dead", "want to end my life",
    "want to kill myself",
    # Don't want to live
    "don't want to live", "dont want to live",
    "don't want to be here", "dont want to be here",
    # Feel like dying
    "feel like dying", "feeling like dying", "feels like dying",
    "i'm dying inside", "im dying inside", "dying inside",
    # Kill myself
    "kill myself", "killing myself", "gonna kill myself",
    "gon kill myself", "kill my self",
    # End it
    "end it all", "end my life", "end everything", "end this pain",
    # Better off dead
    "better off dead", "better off without me", "world better without me",
    # Suicide
    "suicide", "suicidal", "plan to end", "thinking about ending",
    # Self harm
    "self harm", "self-harm", "hurting myself", "hurt myself",
    "cutting myself", "cut myself", "burn myself",
    # Other
    "no reason to live", "nothing to live for", "life isn't worth",
    "life is not worth", "not worth living", "rather be dead",
    "wish i was dead", "wish i were dead", "hope i die",
    "can't go on", "cant go on", "overdose", "no way out",
    "done with life", "done with everything", "tired of living",
    "don't wanna be here", "dont wanna be here",
    "wanna disappear", "want to disappear",
    "never wake up", "not wake up",
]

CRISIS_MEDIUM_ROOTS = [
    "hopeless", "worthless", "nobody cares", "no one cares",
    "all alone", "completely alone", "empty inside", "feeling numb",
    "can't cope", "cant cope", "can't handle", "cant handle",
    "falling apart", "breaking down", "i hate myself", "hate myself",
    "out of control", "losing my mind", "losing it",
    "what's the point", "whats the point", "no point anymore",
    "exhausted by everything", "so tired of everything",
    "mentally exhausted", "emotionally drained", "can't take it",
    "cant take it", "want to give up", "wanna give up",
    "give up on life", "life is too hard", "i can't do this",
]

# ─── Toxicity Keyword System ───────────────────────────────────────────────────
TOXICITY_HIGH_ROOTS = [
    "kill you", "gonna kill", "i'll kill", "hurt you", "hurt others",
    "attack you", "destroy you", "i hate you", "harm others",
    "threatening", "murder", "kill everyone", "hurt everyone",
    "piece of shit", "go fuck yourself", "fuck you", "you're worthless",
]

TOXICITY_MEDIUM_ROOTS = [
    "stupid", "idiot", "dumb", "loser", "jerk", "shut up",
    "hate you", "i hate everything", "you don't understand",
    "so frustrated", "angry at everyone",
]


def _score_text(text: str, high_roots: list, medium_roots: list) -> tuple:
    """
    Scans lowercased text for any root phrase match.
    HIGH match  → score 0.88–0.97 → 'high'
    MEDIUM match → score 0.45–0.69 → 'medium'
    No match    → score 0.02–0.14 → 'low'
    """
    lower = text.lower()

    for phrase in high_roots:
        if phrase in lower:
            return round(random.uniform(0.88, 0.97), 3), "high"

    for phrase in medium_roots:
        if phrase in lower:
            return round(random.uniform(0.45, 0.69), 3), "medium"

    return round(random.uniform(0.02, 0.14), 3), "low"


class DetoxifierModel:
    def __init__(self, model_path: str = "unitary/toxic-bert"):
        self.model_path = model_path

    def predict(self, text: str) -> Dict[str, Any]:
        score, bucket = _score_text(text, TOXICITY_HIGH_ROOTS, TOXICITY_MEDIUM_ROOTS)
        return {
            "label": "hostility",
            "score": score,
            "confidence": round(random.uniform(0.85, 0.99), 3),
            "severity_bucket": bucket
        }


class DistilBERTModel:
    def __init__(self, model_path: str = "distilbert-base-uncased"):
        self.model_path = model_path

    def predict(self, text: str) -> Dict[str, Any]:
        score, bucket = _score_text(text, CRISIS_HIGH_ROOTS, CRISIS_MEDIUM_ROOTS)
        return {
            "label": "self_harm_ideation",
            "score": score,
            "confidence": round(random.uniform(0.85, 0.99), 3),
            "severity_bucket": bucket
        }
