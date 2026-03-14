import streamlit as st
from src.privacy import PresidioPrivacyLayer
from src.safety_models import DetoxifierModel, DistilBERTModel
from src.guardrails import evaluate_guardrails
from src.vector_store import CBDRetrievalLayer
from src.llm_engine import MiddleLLMEngine

@st.cache_resource(show_spinner="Loading Privacy Layer (Presidio)...")
def load_privacy_layer():
    return PresidioPrivacyLayer()

# Safety models are lightweight heuristic classifiers — no heavy weights to load.
# Do NOT cache them so code changes (keyword lists) always take effect immediately.
def load_detoxifier():
    return DetoxifierModel()

def load_distilbert():
    return DistilBERTModel()

@st.cache_resource(show_spinner="Loading CBD Technique Vector Store...")
def load_cbd_store():
    return CBDRetrievalLayer()

@st.cache_resource(show_spinner="Connecting to Qwen2.5-72B LLM Engine...")
def load_llm_engine():
    return MiddleLLMEngine()

privacy_layer = load_privacy_layer()
detoxifier    = load_detoxifier()
distilbert    = load_distilbert()
cbd_store     = load_cbd_store()
middle_llm    = load_llm_engine()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="MindMate", page_icon="🧠", layout="wide")
st.title("🧠 MindMate — Mental Health Support")
st.caption("A privacy-aware, empathetic assistant with safety guardrails. Not a replacement for professional help.")

# ─── Session State Initialisation ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []        # Full chat history
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""   # Rolling LLM summary

# ─── Sidebar: Conversation Summary ────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Conversation Summary")
    if st.session_state.conversation_summary:
        st.info(st.session_state.conversation_summary)
    else:
        st.caption("Summary will appear after a few messages.")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_summary = ""
        st.rerun()

# ─── Display Existing Chat History ────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Handle New User Input ────────────────────────────────────────────────────
if prompt := st.chat_input("How are you feeling today?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("MindMate is thinking..."):
        # 1. Privacy Layer — strip PII
        sanitized_message = privacy_layer.sanitize(prompt)

        # 2. Safety Scoring
        tox_result    = detoxifier.predict(sanitized_message)
        crisis_result = distilbert.predict(sanitized_message)

        # 3. Guardrails evaluation
        guardrails_activated, escalation_msg = evaluate_guardrails(tox_result, crisis_result)

        if guardrails_activated:
            response_text = escalation_msg
            status   = "🚨 Escalated — Guardrails Activated"
            used_cbd = "None"
        else:
            # 4. CBD Vector Retrieval — run against the SANITIZED text
            relevant_cbd = cbd_store.retrieve(sanitized_message, top_k=2)
            used_cbd     = [t['name'] for t in relevant_cbd]

            # 5. Generate via Qwen2.5-72B with conversation summary as context
            response_text = middle_llm.generate(
                sanitized_input=sanitized_message,
                cbd_techniques=relevant_cbd,
                toxicity_level=tox_result['severity_bucket'],
                crisis_level=crisis_result['severity_bucket'],
                conversation_summary=st.session_state.conversation_summary
            )
            status = "✅ Safe — LLM Generated"

    # 6. Append assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # 7. Rolling summary — regenerate every 4 full turns (8 messages)
    if len(st.session_state.messages) % 8 == 0:
        with st.spinner("Updating conversation summary..."):
            st.session_state.conversation_summary = middle_llm.summarize_conversation(
                st.session_state.messages
            )

    # 8. Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)

    # 9. System Trace expander
    with st.expander("🔍 System Trace (Pipeline Details)"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔒 After Presidio (PII Removed):**")
            st.code(sanitized_message, language="text")
            st.markdown("**🛡️ Toxicity Score (Detoxifier):**")
            st.json(tox_result)
        with col2:
            st.markdown("**🚨 Crisis Score (DistilBERT):**")
            st.json(crisis_result)
            st.markdown("**🧠 CBD Techniques Retrieved:**")
            for t in (relevant_cbd if not guardrails_activated else []):
                st.markdown(f"- **{t['name']}**: {t['description']}")
        st.markdown(f"**Pipeline Status:** {status}")
