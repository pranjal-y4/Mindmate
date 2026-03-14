# MindMate — As-Built System Architecture

> **Last Updated:** March 2026  
> This document reflects the actual implemented system, not the original design brief.

---

## Overview

MindMate is a privacy-aware, empathetic mental health support assistant built as a multi-layer NLP inference pipeline. Every user message passes through a strict sequence of safety, privacy, and generative AI components before a response is produced. The system is designed to be grounded, non-hallucinating, and safe-by-default, with automatic escalation when crisis language is detected.

**What it is:** A supportive conversational tool using retrieval-augmented generation, heuristic safety classifiers, CBT-inspired techniques, and a free open-weights LLM.  
**What it is not:** A licensed therapist, crisis service, or medical tool.

---

## Architecture Diagram (Text)

```
User Input (Raw Text)
        │
        ▼
┌──────────────────────────────┐
│   1. Presidio Privacy Layer  │  ← PII removal (names, emails, phones, locations)
│      (Microsoft Presidio +   │
│       spaCy en_core_web_lg)  │
└──────────────┬───────────────┘
               │ Sanitized Text
               ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│   2a. Detoxifier Model       │    │   2b. DistilBERT Crisis Model │
│   (Heuristic root-phrase     │    │   (Heuristic root-phrase      │
│    classifier)               │    │    classifier)                │
│   Output: toxicity score,    │    │   Output: crisis score,       │
│   severity: low/medium/high  │    │   severity: low/medium/high   │
└──────────────┬───────────────┘    └──────────────┬────────────────┘
               └────────────┬───────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   3. Guardrails Engine  │
              │   (Rule-based router)   │
              └──────────┬──────────────┘
                 ┌───────┴────────┐
          TRIGGERED?            NOT TRIGGERED
                 │                    │
                 ▼                    ▼
     Return Safety Message    ┌───────────────────────┐
     (crisis/de-escalation)   │  4. CBD Retrieval      │
                              │  (FAISS + sentence-    │
                              │   transformers ANN)    │
                              └───────────┬────────────┘
                                          │ Top-2 techniques
                                          ▼
                              ┌───────────────────────┐
                              │  5. Prompt Assembly    │
                              │  System rules +        │
                              │  safety context +      │
                              │  conversation summary  │
                              │  + CBD techniques +    │
                              │  sanitized input       │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌───────────────────────────┐
                              │  6. Qwen2.5-72B-Instruct  │
                              │  (HuggingFace Serverless  │
                              │   Inference API — free)   │
                              └───────────┬───────────────┘
                                          │
                                          ▼
                                   Final Response
```

---

## Components

### 1. Privacy Layer — `src/privacy.py`
- **Library:** Microsoft Presidio (`presidio-analyzer` + `presidio-anonymizer`)
- **NLP Model:** spaCy `en_core_web_lg` (400MB, loaded once via `@st.cache_resource`)
- **Entities Detected:** PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, URL, NRP, IBAN_CODE, CRYPTO, US_SSN, and more
- **Mechanism:** `AnalyzerEngine` detects entity spans → `AnonymizerEngine` replaces them with typed tokens (e.g., `<PERSON>`, `<EMAIL_ADDRESS>`)
- **Used at:** Both training-time (dataset anonymization) and inference-time (every live request)

### 2a. Detoxifier Model — `src/safety_models.py` → `DetoxifierModel`
- **Architecture:** Heuristic root-phrase classifier (production path: `unitary/toxic-bert`)
- **Mechanism:** Scans lowercased sanitized text against two ranked phrase lists using Python `in` operator
- **Target Signals:** Abusive language, harassment, hate, threats directed outward, hostile escalation
- **Output Schema:**
  ```json
  { "label": "hostility", "score": 0.92, "confidence": 0.96, "severity_bucket": "high" }
  ```
- **Severity Mapping:**
  - High phrase match → score 0.88–0.97 → `"high"`
  - Medium phrase match → score 0.45–0.69 → `"medium"`
  - No match → score 0.02–0.14 → `"low"`

### 2b. DistilBERT Crisis Model — `src/safety_models.py` → `DistilBERTModel`
- **Architecture:** Heuristic root-phrase classifier (production path: `distilbert-base-uncased` fine-tuned)
- **Mechanism:** Same `_score_text()` engine as Detoxifier with a separate phrase list covering crisis language
- **Target Signals:** Self-harm ideation, suicidal intent, hopelessness, harm-to-others, severe distress
- **Key Phrase Coverage (60+ entries):** "wanna die", "want to die", "feel like dying", "kill myself", "wanna disappear", "no reason to live", "suicide", "self harm", "cutting myself", "better off dead", and many more natural-language variants
- **Output Schema:**
  ```json
  { "label": "self_harm_ideation", "score": 0.91, "confidence": 0.93, "severity_bucket": "high" }
  ```
- **Important:** These models are **not cached** in Streamlit to ensure phrase list updates take effect immediately

### 3. Guardrails Engine — `src/guardrails.py`
A deterministic rule-based decision router. No ML involved — pure conditional logic for reliability.

| Toxicity | Crisis | Action |
|---|---|---|
| HIGH | HIGH | Hard escalation: emergency message + crisis line |
| LOW/MED | HIGH | Soft escalation: crisis support message + encourage professional contact |
| HIGH | LOW/MED | De-escalation: set conversational boundary, redirect gently |
| LOW/MED | LOW/MED | Pass-through: proceed to CBD retrieval + LLM generation |

### 4. CBD Retrieval Layer — `src/vector_store.py`
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (80MB, loaded once)
- **Vector Index:** `faiss.IndexFlatL2` (exact L2 nearest-neighbour search across 10 vectors)
- **Indexed Documents:** 10 structured CBT-inspired coping techniques:
  1. Reframing
  2. Grounding (5-4-3-2-1)
  3. Breathing (Box Breathing)
  4. Journaling
  5. Thought Observation
  6. Behavior Activation
  7. Self-compassion Prompts
  8. Distress Tolerance (TIPP)
  9. Emotion Labeling
  10. Coping Statements
- **Retrieval:** `top_k=2` most semantically similar techniques per query
- **Purpose:** Grounds the LLM in concrete, structured coping strategies; reduces hallucinations

### 5. LLM Engine — `src/llm_engine.py`
- **Model:** `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Serverless Inference API (free tier)
- **API Call:** `InferenceClient.chat_completion()` with `max_tokens=400`, `temperature=0.75`
- **Authentication:** `HF_TOKEN` environment variable (set in Streamlit secrets for deployment)
- **Prompt Design:** Context-aware system prompt injected with:
  - Core personality rules (empathetic, non-judgmental, motivating, non-diagnostic)
  - Conversation summary (rolling, regenerated every 8 messages)
  - Safety status (toxicity + crisis buckets)
  - Retrieved CBD techniques (natural language instructions on how to weave them in)
  - Situation-specific guidance (casual vs. distress vs. crisis path)
- **Fallback:** Intelligent mock responses if API is unavailable, differentiated by greeting/distress/crisis

### 6. Conversation Summary — `middle_llm.summarize_conversation()`
- Triggered every 8 messages (4 full turns)
- Makes a separate `chat_completion()` call to Qwen2.5-72B with a summarization system prompt
- Produces a 2–3 sentence neutral summary of the user's emotional state, topics discussed, and coping strategies mentioned
- Injected into every subsequent LLM call as `CONVERSATION CONTEXT`
- Displayed live in the Streamlit sidebar

---

## Data Flow: Reddit → Training

```
Reddit Mental Health Communities (via `pro` library)
        │
        ▼
reddit_raw.json  (posts + comments + metadata)
        │
        ▼
Presidio Anonymizer (batch offline PII removal)
        │
        ▼
reddit_cleaned.json  (anonymized, preprocessing_status="anonymized")
        │
        ▼
Manual/Semi-automatic labelling:
  - crisis_label: low / medium / high
  - toxicity_label: low / medium / high
        │
        ▼
HuggingFace datasets.Dataset
        │
        ▼
DistilBertForSequenceClassification (fine-tune, 3 classes)
        │
        ▼
Saved weights → models/mindmate-distilbert-crisis/
```

Training script: `src/train.py` (uses HuggingFace `Trainer` + `TrainingArguments`)

---

## Inference Pipeline (Per Request)

```
1. Receive raw user message
2. privacy_layer.sanitize(text)               → PII stripped
3. detoxifier.predict(sanitized)              → tox_result (score, bucket)
4. distilbert.predict(sanitized)              → crisis_result (score, bucket)
5. evaluate_guardrails(tox, crisis)           → (activated, escalation_msg)
   IF activated: return escalation_msg immediately
6. cbd_store.retrieve(sanitized, top_k=2)     → relevant_cbd [list of 2]
7. middle_llm.generate(                       → LLM response
       sanitized, cbd, tox_level,
       crisis_level, conversation_summary)
8. Append to session history
9. Every 8 messages: regenerate conversation_summary
10. Return response to user
```

---

## Application Layer — `app.py`

- **Framework:** Streamlit
- **Caching Strategy:**
  - `@st.cache_resource`: Presidio, CBDRetrievalLayer, MiddleLLMEngine (heavy — load once)
  - **No cache**: DetoxifierModel, DistilBERTModel (lightweight — must always reflect latest phrase lists)
- **Features:**
  - Chat interface with history
  - Sidebar with live rolling conversation summary
  - Clear Chat button
  - System Trace expander showing: Presidio output, toxicity JSON, crisis JSON, CBD techniques retrieved, pipeline status

---

## Deployment

- **Platform:** Streamlit Community Cloud (`share.streamlit.io`)
- **Entry Point:** `app.py`
- **Required Secrets:** `HF_TOKEN` (set in Streamlit Cloud Secrets)
- **Config:** `.streamlit/config.toml`
- **Dependencies:** `requirements.txt`

---

## Risks and Limitations

| Risk | Mitigation |
|---|---|
| Heuristic crisis detection misses novel phrasing | Phrase list is regularly extended; long-term: replace with fine-tuned DistilBERT |
| HF free API rate limits | Intelligent mock fallback returns context-aware responses if API fails |
| Presidio misses hyper-localised PII | Accepted limitation; users are warned not to share identifying information |
| LLM may drift from safety instructions | Prompt includes explicit prohibitions; guardrails act as a hard upstream filter |
| Conversation summary lag | Summary only updates every 8 messages; very early context may be missing |
| Not a real crisis service | Escalation message includes Samaritans (116 123); prominent disclaimer on UI |
