import streamlit as st
import torch
import pickle
from phpcodegen import PHPGenerator, tokenize, generate_php_code

# Load vocab
with open('vocab.pkl', 'rb') as f:
    vocab, reverse_vocab = pickle.load(f)

# Load model
model = PHPGenerator()
model.load_state_dict(torch.load('php_generator.pth'))
model.eval()

st.title("🚀 PHP Code Generator AI")
st.markdown("Describe what you want your PHP code to do, and the AI will generate it for you!")

description = st.text_input("Enter a description of the PHP functionality you need:", 
                           placeholder="e.g., connect to mysql database")

if st.button("Generate PHP Code"):
    if description.strip():
        with st.spinner("Generating code..."):
            generated_code = generate_php_code(description, model, vocab, reverse_vocab)
        
        st.success("Code generated!")
        st.code(generated_code, language='php')
        
        # Copy button (optional)
        st.text_area("Copy the code:", value=generated_code, height=200)
    else:
        st.warning("Please enter a description.")

st.markdown("---")
st.markdown("**Examples to try:**")
st.markdown("- connect to mysql database")
st.markdown("- validate email address")
st.markdown("- read file content")
st.markdown("- create login form")