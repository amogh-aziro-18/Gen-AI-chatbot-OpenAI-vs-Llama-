import streamlit as st
import os
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# -----------------------------
# 🌟 Setup
# -----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
st.set_page_config(page_title="Q&A Multi-Model Chatbot", layout="wide")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_MAP = {
    "GPT-4o-mini (OpenAI)": "openai/gpt-4o-mini",
    "GPT-4o (OpenAI)": "openai/gpt-4o",
    "LLaMA-3 (8B Instruct)": "meta-llama/llama-3-8b-instruct",
    "LLaMA-3 (13B Instruct)": "meta-llama/llama-3-13b-instruct",
}

# -----------------------------
# 🧩 Memory (FAISS-like CPU)
# -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = []  # [(prompt, responses_dict, embedding)]
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # [(role, content)]

def get_embedding(text):
    return np.array(embedder.encode(text))

def store_memory(prompt, responses):
    emb = get_embedding(prompt)
    st.session_state.memory.append((prompt, responses, emb))

def retrieve_context(query, top_k=3):
    if not st.session_state.memory:
        return []
    q_emb = get_embedding(query)
    sims = [cosine_similarity([q_emb], [m[2]])[0][0] for m in st.session_state.memory]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [st.session_state.memory[i] for i in top_idx]

# -----------------------------
# 🎨 UI Header
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#00FFFF; text-shadow:0 0 15px #00FFFF;'>
    ⚡ Q&A Multi-Model Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)

# Sidebar history
with st.sidebar:
    st.title("🧠 Chat History")
    if st.session_state.memory:
        for i, (prompt, responses, _) in enumerate(st.session_state.memory):
            with st.expander(f"💬 {prompt[:60]}..."):
                st.markdown(f"**You:** {prompt}")
                for model_name, resp in responses.items():
                    st.markdown(f"**{model_name}:** {resp}")
    else:
        st.info("No previous chats yet!")

# -----------------------------
# 🔀 Mode Selection
# -----------------------------
mode = st.radio("Choose Mode:", ["Single Model Chat", "Side-by-Side Comparison"], horizontal=True)

if "single_model_choice" not in st.session_state:
    st.session_state.single_model_choice = list(MODEL_MAP.keys())[0]
if "compare_model_a" not in st.session_state:
    keys = list(MODEL_MAP.keys())
    st.session_state.compare_model_a = keys[0]
if "compare_model_b" not in st.session_state:
    keys = list(MODEL_MAP.keys())
    st.session_state.compare_model_b = keys[1] if len(keys) > 1 else keys[0]

# Model selectors
if mode == "Single Model Chat":
    st.markdown("<h4 style='color:#00FFFF;'>🤖 Choose your model:</h4>", unsafe_allow_html=True)
    st.session_state.single_model_choice = st.selectbox(
        "Select model for chat:",
        list(MODEL_MAP.keys()),
        index=list(MODEL_MAP.keys()).index(st.session_state.single_model_choice)
    )
else:
    st.markdown("<h4 style='color:#00FFFF;'>⚖️ Compare two models:</h4>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.compare_model_a = st.selectbox(
            "Model A:",
            list(MODEL_MAP.keys()),
            index=list(MODEL_MAP.keys()).index(st.session_state.compare_model_a),
            key="sel_model_a"
        )
    with c2:
        st.session_state.compare_model_b = st.selectbox(
            "Model B:",
            list(MODEL_MAP.keys()),
            index=list(MODEL_MAP.keys()).index(st.session_state.compare_model_b),
            key="sel_model_b"
        )

st.markdown("---")

# -----------------------------
# ✏️ Prompt Input
# -----------------------------
st.markdown("<h4 style='color:#00FFFF;'>💡 Enter your prompt below:</h4>", unsafe_allow_html=True)
prompt = st.text_area("Prompt", placeholder="Type your question here...", height=120)

col1, col2 = st.columns([1, 5])
with col1:
    generate = st.button("🚀 Generate")
with col2:
    clear = st.button("🧹 Clear Memory")

if clear:
    st.session_state.memory = []
    st.session_state.conversation = []
    st.experimental_rerun()

# -----------------------------
# ⚙️ Request Function
# -----------------------------
def get_openrouter_response(model_id, prompt, context=""):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    # Different system prompts per model type
    if "openai" in model_id:
        system_prompt = (
            "You are ChatGPT — a factual and structured AI assistant by OpenAI. "
            "Answer in a professional, academic tone using Markdown formatting. "
            "Always use clear sections such as **Overview**, **Key Details**, **Notable Points**, and **Summary**. "
            "Use bullet points or numbered lists for clarity. "
            "Do not invent information — if uncertain, state so clearly."
        )
    elif "llama" in model_id:
        system_prompt = (
            "You are LLaMA — a helpful and logical AI model. "
            "Give concise, factual answers similar to ChatGPT’s style. "
            "Use headings (###), bullet points, and structured paragraphs. "
            "Avoid hallucinations, irrelevant text, or non-English content. "
            "If you are unsure of details, write: 'I am not fully certain about this point.'"
        )
    else:
        system_prompt = (
            "You are a reliable AI assistant. Always provide factual, well-structured, Markdown-formatted answers."
        )

    # Build messages
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context:\n{context}"} if context else {},
            {"role": "user", "content": prompt}
        ]
    }
    data["messages"] = [m for m in data["messages"] if m]

    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
        r.raise_for_status()
        resp = r.json()
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error: {e}"



# -----------------------------
# 💬 Chat Logic
# -----------------------------
if generate and prompt.strip():
    st.markdown("### ✨ Generating response...")
    context_items = retrieve_context(prompt, top_k=3)
    context_text = "\n".join([f"Previous Q: {p}\nA: {list(r.values())[0]}" for p, r, _ in context_items])

    if mode == "Single Model Chat":
        model_display = st.session_state.single_model_choice
        model_id = MODEL_MAP[model_display]
        response = get_openrouter_response(model_id, prompt, context_text)

        st.session_state.conversation.append(("user", prompt))
        st.session_state.conversation.append(("assistant", response))
        store_memory(prompt, {model_display: response})

        st.markdown("### 💬 Conversation")
        for role, text in st.session_state.conversation:
            color = "#00FFFF" if role == "assistant" else "#FFFFFF"
            bg = "#071029" if role == "assistant" else "#111"
            st.markdown(
                f"<div style='background:{bg}; padding:10px; border-radius:10px; margin:6px 0; color:{color};'><b>{'Bot' if role=='assistant' else 'You'}:</b> {text}</div>",
                unsafe_allow_html=True,
            )

    else:
        model_a = st.session_state.compare_model_a
        model_b = st.session_state.compare_model_b
        model_a_id = MODEL_MAP[model_a]
        model_b_id = MODEL_MAP[model_b]

        gpt_resp = get_openrouter_response(model_a_id, prompt, context_text)
        llama_resp = get_openrouter_response(model_b_id, prompt, context_text)
        store_memory(prompt, {model_a: gpt_resp, model_b: llama_resp})

        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"### 🧠 {model_a}")
            st.markdown(f"<div style='background:#0b0b17; padding:12px; border-radius:10px; color:#00FFFF;'>{gpt_resp}</div>", unsafe_allow_html=True)
        with colB:
            st.markdown(f"### 🦙 {model_b}")
            st.markdown(f"<div style='background:#0b0b17; padding:12px; border-radius:10px; color:#00FFFF;'>{llama_resp}</div>", unsafe_allow_html=True)

# -----------------------------
# 🦋 Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>⚡ Built by <b style='color:#00FFFF;'>Amogh</b></p>",
    unsafe_allow_html=True
)
