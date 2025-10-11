import streamlit as st
import pandas as pd
import base64
import io
from PIL import Image
import pydicom
from openai import OpenAI
from huggingface_hub import InferenceClient
from transformers import pipeline
import time
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt
from datetime import datetime
import psutil
from cryptography.fernet import Fernet

# Generate or load encryption key (in production, store securely or let user provide)
if 'encryption_key' not in st.session_state:
    st.session_state.encryption_key = Fernet.generate_key()
cipher = Fernet(st.session_state.encryption_key)

# Session state for persistence
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = 0
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {}
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 5

st.title("Med-Harmonizer: MRI Analysis AI")
st.warning("LLM outputs are ~60-80% accurate per 2025 benchmarks; consult a radiologist. Not for clinical use.")

# Step 1: Upload images with pydicom support
uploaded_files = st.file_uploader("Upload MRI Images", type=["png", "jpg", "dcm"], accept_multiple_files=True)
if uploaded_files:
    successful_uploads = []
    for f in uploaded_files:
        try:
            if f.name.endswith('.dcm'):
                ds = pydicom.dcmread(f)
                img_array = ds.pixel_array
                image = Image.fromarray(img_array).convert('RGB')
            else:
                image = Image.open(f)
            successful_uploads.append({'file': f, 'image': image, 'metadata': ds if 'ds' in locals() else None})
        except Exception as e:
            st.error(f"Failed to process {f.name}: {str(e)}")
    st.session_state.uploaded_files = successful_uploads
    st.write(f"Total images uploaded successfully: {len(successful_uploads)}")

# Model list with versions and auth info
models_info = {
    "Grok 4 (xAI)": {"version": "grok-4", "auth": "API Key (from xAI pro account)", "client_type": "xai", "model": "grok-4"},
    "GPT-5 (OpenAI)": {"version": "gpt-5", "auth": "API Key (from OpenAI pro/Plus)", "client_type": "openai", "model": "gpt-5"},
    "DeepSeek V3.2-Exp": {"version": "V3.2-Exp", "auth": "API Key (text-only; using BLIP captioner)", "client_type": "deepseek", "model": "deepseek-v3.2-exp"},
    "LLaVA (Hugging Face)": {"version": "1.5-7B", "auth": "Optional HF Token (free tier)", "client_type": "hf", "model": "llava-hf/llava-1.5-7b-hf"},
    "MedGemma (Google/HF)": {"version": "4B Multimodal", "auth": "Optional HF Token", "client_type": "hf", "model": "google/medgemma-4b-multimodal"},
}

# Step 2: Select models (max 3 checkboxes) with clear button
if st.session_state.uploaded_files:
    st.subheader("Select up to 3 Models for Analysis")
    selected = []
    cols = st.columns(2)
    for i, (model_name, info) in enumerate(models_info.items()):
        with cols[i % 2]:
            if st.checkbox(f"{model_name} - Version: {info['version']} | Auth: {info['auth']}", key=model_name):
                selected.append(model_name)
                if len(selected) > 3:
                    st.warning("Max 3 models allowed. Deselect one.")
                    break
    st.session_state.selected_models = selected[:3]

    if st.button("Clear Model Selections"):
        st.session_state.selected_models = []
        st.rerun()

    # User input for API keys
    st.subheader("Enter API Keys")
    openai_key = st.text_input("OpenAI API Key (for GPT-5)", type="password", key="openai_key")
    hf_token = st.text_input("Hugging Face Token (for LLaVA/MedGemma)", type="password", key="hf_token")
    encryption_key = st.text_input("Encryption Key (base64-encoded)", type="password", key="encryption_key")

    # Encrypt and store keys if provided
    if openai_key:
        st.session_state.api_keys['OPENAI_API_KEY'] = cipher.encrypt(openai_key.encode())
    if hf_token:
        st.session_state.api_keys['HF_TOKEN'] = cipher.encrypt(hf_token.encode())
    if encryption_key:
        try:
            st.session_state.encryption_key = base64.b64decode(encryption_key)
            cipher = Fernet(st.session_state.encryption_key)
            st.success("Encryption key updated successfully.")
        except Exception as e:
            st.error(f"Invalid encryption key: {str(e)}")

# Fixed structured prompt
fixed_prompt = """
Interpret this MRI scan. Output in JSON format:
{
  "abnormalities": "Description of any abnormalities",
  "diagnoses": "Potential diagnoses",
  "confidence": "Confidence level (0-100)"
}
"""

# BLIP captioner for text-only models
@st.cache_resource
def get_captioner():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Memory check
if psutil.virtual_memory().percent > 80:
    st.warning("High memory usage; reduce batch size or close other apps.")

# Step 3: Configurable batch size and process
if st.session_state.selected_models and st.session_state.uploaded_files:
    st.session_state.batch_size = st.slider("Batch Size", 1, 10, st.session_state.batch_size)
    total_images = len(st.session_state.uploaded_files)
    start_idx = st.session_state.current_batch * st.session_state.batch_size
    end_idx = min(start_idx + st.session_state.batch_size, total_images)
    batch_files = st.session_state.uploaded_files[start_idx:end_idx]

    if st.button(f"Analyze Batch {st.session_state.current_batch + 1} ({start_idx+1}-{end_idx} of {total_images})"):
        progress = st.progress(0)
        with st.spinner("Processing batch..."):

            @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
            async def analyze_image(item, model_name):
                try:
                    image = item['image']
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='JPEG')
                    base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

                    info = models_info[model_name]
                    encrypted_key = st.session_state.api_keys.get(f"{info['client_type'].upper()}_API_KEY" if info['client_type'] in ['openai', 'xai'] else f"{info['client_type'].upper()}_TOKEN", b'')
                    api_key = cipher.decrypt(encrypted_key).decode() if encrypted_key else ""

                    if info['client_type'] == "openai":
                        client = OpenAI(api_key=api_key)
                        response = client.chat.completions.create(
                            model=info['model'],
                            messages=[{"role": "user", "content": [{"type": "text", "text": fixed_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
                        )
                        output = response.choices[0].message.content
                    elif info['client_type'] == "xai":
                        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
                        response = client.chat.completions.create(
                            model=info['model'],
                            messages=[{"role": "user", "content": [{"type": "text", "text": fixed_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
                        )
                        output = response.choices[0].message.content
                    elif info['client_type'] == "hf":
                        client = InferenceClient(token=api_key or None)
                        response = client.post(json={"inputs": fixed_prompt, "image": base64_image}, model=info['model'])
                        output = response.json()[0].get("generated_text", "Error")
                    elif info['client_type'] == "deepseek":
                        captioner = get_captioner()
                        caption = captioner(base64_image)[0]['generated_text']
                        prompt_with_caption = f"{fixed_prompt} based on this description: {caption}"
                        output = "DeepSeek output (simulated with caption)"  # Replace with actual DeepSeek client
                    return {f"{model_name} Output": output, f"{model_name} Version": info['version']}
                except Exception as e:
                    if "rate_limit" in str(e):
                        raise
                    return {f"{model_name} Output": f"Error: {str(e)}"}

            async def process_batch():
                tasks = []
                for i, item in enumerate(batch_files):
                    row = {'Image File Name': item['file'].name, 'Timestamp': datetime.now()}
                    for model in st.session_state.selected_models:
                        tasks.append(analyze_image(item, model))
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for res in results:
                        if not isinstance(res, Exception):
                            row.update(res)
                    st.session_state.results_df = pd.concat([st.session_state.results_df, pd.DataFrame([row])], ignore_index=True)
                    progress.progress((i + 1) / len(batch_files))
                    await asyncio.sleep(0)

            asyncio.run(process_batch())

        st.success("Batch processed!")
        st.dataframe(st.session_state.results_df)

        # Incremental download after batch
        csv = st.session_state.results_df.to_csv(index=False)
        st.download_button("Download Partial Results as CSV", csv, "mri_partial_results.csv", "text/csv")

    # Step 4: Ask for next batch
    if end_idx < total_images:
        if st.button("Analyze Next Batch?"):
            st.session_state.current_batch += 1
            st.rerun()
    else:
        st.success("All images processed!")

# Final download
if not st.session_state.results_df.empty:
    csv = st.session_state.results_df.to_csv(index=False)
    st.download_button("Download Full Results as CSV", csv, "mri_results.csv", "text/csv")
