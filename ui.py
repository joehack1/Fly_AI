import streamlit as st
import torch
import pickle
import time
from phpcodegen import PHPGenerator, tokenize, generate_php_code

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="PHP Code Generator AI",
    page_icon="🚀",
    layout="centered"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 PHP Generator AI")
st.sidebar.markdown("""
Generate PHP code from plain English descriptions.

**Powered by PyTorch**
""")

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tips**")
st.sidebar.markdown("- Be specific")
st.sidebar.markdown("- One task at a time")
st.sidebar.markdown("- Simple English works best")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open('vocab.pkl', 'rb') as f:
        vocab, reverse_vocab = pickle.load(f)

    model = PHPGenerator()
    model.load_state_dict(torch.load('php_generator.pth', map_location="cpu"))
    model.eval()

    return model, vocab, reverse_vocab

model, vocab, reverse_vocab = load_model()

# ---------------- MAIN UI ----------------
st.title("🚀 PHP Code Generator AI")
st.caption("Describe PHP functionality. Watch code come to life.")

st.markdown("")

description = st.text_input(
    "📝 What should the PHP code do?",
    placeholder="e.g. connect to MySQL database using PDO"
)

generate_btn = st.button("✨ Generate PHP Code", use_container_width=True)

# ---------------- TYPING EFFECT FUNCTION ----------------
def type_writer(text, speed=0.01):
    placeholder = st.empty()
    typed = ""

    for char in text:
        typed += char
        placeholder.code(typed, language="php")
        time.sleep(speed)

# ---------------- GENERATION ----------------
if generate_btn:
    if not description.strip():
        st.warning("Please enter a description.")
    else:
        with st.spinner("🧠 Thinking like a PHP developer..."):
            time.sleep(0.5)
            generated_code = generate_php_code(
                description, model, vocab, reverse_vocab
            )

        st.success("✅ Code generated")

        # Typing animation
        type_writer(generated_code, speed=0.005)

        # Copy area
        st.markdown("### 📋 Copy Code")
        st.text_area(
            "PHP Code",
            value=generated_code,
            height=220
        )

# ---------------- EXAMPLES ----------------
st.markdown("---")
st.markdown("### 🧪 Example prompts")
cols = st.columns(2)

examples = [
    "connect to mysql database",
    "validate email address",
    "read file content",
    "create login form",
    "hash password",
    "upload file securely"
]

for i, example in enumerate(examples):
    cols[i % 2].markdown(f"- `{example}`")
