# 🧠 MindMate: Privacy-Preserving Mental Health Assistant

MindMate is a safe, empathetic, and privacy-aware mental health support assistant. It uses a multi-layered NLP pipeline to provide supportive conversations while ensuring user data is anonymized and high-distress situations are handled with immediate safe guardrails.

---

## ✨ Key Features

- **🔒 Real-time Privacy (Presidio):** Automatically detects and masks Personally Identifiable Information (PII) like names, emails, and locations before any data is processed by the AI.
- **🛡️ Safety Guardrails:** Uses a custom heuristic classification engine to detect toxicity and crisis language (e.g., self-harm ideation) with high sensitivity.
- **🧠 Grounded Support (RAG):** Integrates 10 structured Cognitive Behavioral Therapy (CBT) inspired techniques, retrieving the most relevant ones to guide the conversation.
- **💬 Empathetic LLM:** Powered by **Qwen2.5-72B-Instruct** for warm, natural, and motivating responses that validate feelings before suggesting coping strategies.
- **📋 Conversation Memory:** Generates a rolling summary of the chat every few messages to maintain context and provide a personalized experience.
- **🔍 Pipeline Transparency:** A "System Trace" feature allows you to see exactly how your message was sanitized, scored, and which techniques were used under the hood.

---

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.11+
- [Git](https://git-scm.com/)

### 2. Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/pranjal-y4/Mindmate.git
cd Mindmate
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables
You need a [HuggingFace Access Token](https://huggingface.co/settings/tokens) for the LLM engine:
```bash
export HF_TOKEN="your_token_here"
```

### 4. Launch the App
```bash
streamlit run app.py
```

---

## 🛠️ Architecture

MindMate follows a strict 6-step inference pipeline:
1. **Privacy Layer:** Strip PII using Microsoft Presidio + spaCy.
2. **Safety Scoring:** Two-stage classification for Toxicity (Detoxifier) and Crisis (DistilBERT).
3. **Guardrails Engine:** Rule-based router that activates emergency messages for high-severity signals.
4. **CBD Retrieval:** Semantic search via FAISS to find relevant coping techniques.
5. **Context Assembly:** Combines the rolling conversation summary, safety scores, and retrieved techniques into a specific system prompt.
6. **LLM Generation:** Final response generation via the HuggingFace Inference API.

For a deep dive, check out the [Full Architecture Documentation](./mindmate_architecture.md).

---

## 📦 Deployment (Streamlit Cloud)

MindMate is optimized for Streamlit Cloud:
- **Python Version:** Forced to 3.11 via `runtime.txt` and `.python-version` for stability.
- **Secrets:** Add your `HF_TOKEN` in the Streamlit Dashboard -> Settings -> Secrets.
- **System Deps:** `packages.txt` ensures C++ libraries for FAISS are installed.

---

## ⚠️ Disclaimer
**MindMate is an AI tool and NOT a replacement for professional clinical care, therapy, or emergency services.** 
If you are in immediate danger or a health crisis, please contact your local emergency services or a dedicated crisis hotline.

- **Samaritans (UK/ROI):** 116 123
- **National Suicide Prevention Lifeline (US):** 988
- **Crisis Text Line:** Text HOME to 741741

---
*Created by [Pranjal](https://github.com/pranjal-y4) - Built for compassionate AI interaction.*
