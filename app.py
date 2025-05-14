import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")

generator = pipeline("text-generation", model="28p07/TinyLlama-1.1B-Chat-v1.0-mental-health-counselling", device="cpu")

# Sentence trimmer
def extract_complete_sentences(text):
    pattern = r'(.*?\.\s*)'
    matches = re.findall(pattern, text)
    return ''.join(matches).strip()

# Streamlit UI
st.title("Mental Health Counsellor")
st.write("Tell me what you're feeling:")

text = st.text_input("Enter your thoughts:")

if text:
    output = generator([{"role": "user", "content": text}], max_new_tokens=128, return_full_text=False)[0]
    output = extract_complete_sentences(output["generated_text"])
    st.write(output)
else:
    st.warning("Please enter some text to begin.")
