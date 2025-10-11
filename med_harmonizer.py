import streamlit as st
import pandas as pd
import base64
import io
import asyncio
from PIL import Image
import pydicom
from openai import OpenAI
from huggingface_hub import InferenceClient
from transformers import pipeline
from datetime import datetime
import psutil
from tenacity import retry, wait_exponential, stop_after_attempt
import plotly.express as px
import json

# Initialize session state
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

# App title and disclaimer
st.title("Med-Harmonizer: MRI Analysis AI")
st.warning("LLM outputs are ~60-80% accurate per 2025 benchmarks; consult a radiologist. Not for clinical use.")

# Model configurations
models_info = {
    "Grok 4 (xAI)": {"version": "grok-4", "auth": "API Key (from xAI pro account)", "client_type": "xai", "model": "grok-4"},
    "GPT-5 (OpenAI)": {"version": "gpt-5", "auth": "API Key (from OpenAI pro/Plus)", "client_type": "openai", "model": "gpt-5"},
    "DeepSeek V3.2-Exp": {"version": "V3.2-Exp", "auth": "API Key (text-only; using BLIP captioner)", "client_type": "deepseek", "model": "deepseek-v3.2-exp"},
    "LLaVA (Hugging Face)": {"version": "1.5-7B", "auth": "Optional HF Token (free tier)", "client_type": "hf", "model": "llava-hf/llava-1.5-7b-hf"},
    "MedGemma (Google/HF)": {"version": "4B Multimodal", "auth": "Optional HF Token", "client_type": "hf", "model": "google/medgemma-4b-multimodal"},
}

# Fixed structured prompt
fixed_prompt = """
Interpret this MRI scan. Output in JSON format:
{
  "abnormalities": "Description of any abnormalities",
  "diagnoses": "Potential diagnoses",
  "confidence": "Confidence level (0-100)"
}
"""

# Cached BLIP captioner
@st.cache_resource
def get_captioner():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Cached API client initialization
@st.cache_resource
def init_client(client_type, api_key):
    if client_type == "openai":
        return OpenAI(api_key=api_key)
    elif client_type == "xai":
        return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    elif client_type == "hf":
        return InferenceClient(token=api_key or None)
    elif client_type == "deepseek":
        return None  # Replace with actual DeepSeek client when available
    return None

# Preprocess image (cached)
@st.cache_data
def preprocess_image(file_data, _file_name):
    try:
        if _file_name.endswith('.dcm'):
            ds = pydicom.dcmread(file_data)
            img_array = ds.pixel_array
            image = Image.fromarray(img_array).convert('RGB')
            metadata = {
                "PatientID": getattr(ds, 'PatientID', 'N/A'),
                "StudyDate": getattr(ds, 'StudyDate', 'N/A'),
                "Modality": getattr(ds, 'Modality', 'N/A')
            }
        else:
            image = Image.open(file_data)
            metadata = None
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return image, base64_image, metadata
    except Exception as e:
        return None, None, f"Error processing {_file_name}: {str(e)}"

# Step 1: Upload images
st.subheader("Upload MRI Images")
uploaded_files = st.file_uploader("Upload MRI Images", type=["png", "jpg", "dcm"], accept_multiple_files=True)
if uploaded_files:
    successful_uploads = []
    for f in uploaded_files:
        image, base64_image, metadata = preprocess_image(f, f.name)
        if image:
            successful_uploads.append({'file': f, 'image': image, 'base64_image': base64_image, 'metadata': metadata})
        else:
            st.error(metadata)  # metadata contains error message if processing failed
    st.session_state.uploaded_files = successful_uploads
    st.write(f"Total images uploaded successfully: {len(successful_uploads)}")

# Step 2: Select models
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

    # API key inputs
    st.subheader("Enter API Keys")
    openai_key = st.text_input("OpenAI API Key (for GPT-5)", type="password", key="openai_key")
    hf_token = st.text_input("Hugging Face Token (for LLaVA/MedGemma)", type="password", key="hf_token")
    if openai_key:
        st.session_state.api_keys['openai'] = openai_key
    if hf_token:
        st.session_state.api_keys['hf'] = hf_token

    # Reset button
    if st.button("Reset All"):
        st.session_state.clear()
        st.rerun()

# Memory warning
if psutil.virtual_memory().percent > 80:
    st.warning("High memory usage detected. Consider reducing batch size or closing other apps.")

# Step 3: Batch processing
if st.session_state.selected_models and st.session_state.uploaded_files:
    total_images = len(st.session_state.uploaded_files)
    # Dynamic batch size based on image count
    max_batch_size = min(10, total_images)
    st.session_state.batch_size = st.slider("Batch Size", 1, max_batch_size, st.session_state.batch_size)
    start_idx = st.session_state.current_batch * st.session_state.batch_size
    end_idx = min(start_idx + st.session_state.batch_size, total_images)
    batch_files = st.session_state.uploaded_files[start_idx:end_idx]

    if st.button(f"Analyze Batch {st.session_state.current_batch + 1} ({start_idx+1}-{end_idx} of {total_images})"):
        progress = st.progress(0)
        status_text = st.empty()
        with st.spinner("Processing batch..."):
            @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
            async def analyze_image(item, model_name):
                try:
                    info = models_info[model_name]
                    api_key = st.session_state.api_keys.get(info['client_type'], "")
                    client = init_client(info['client_type'], api_key)
                    base64_image = item['base64_image']
                    if,var output
                    if info['client_type'] == "openai" or info['client_type'] == "xai":
                        response = client.chat.completions.create(
                            model=info['model'],
                            messages=[{"role": "user", "content": [{"type": "text", "text": fixed_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                        )
                        output = response.choices[0].message.content
                    elif info['client_type'] == "hf":
                        response = client.post(json={"inputs": fixed_prompt, "image": base64_image}, model=info['model'])
                        output = response.json()[0].get("generated_text", "Error")
                    elif info['client_type'] == "deepseek":
                        captioner = get_captioner()
                        caption = captioner(base64_image)[0]['generated_text']
                        prompt_with_caption = f"{fixed_prompt} based on this description: {caption}"
                        output = json.dumps({"abnormalities": "Simulated", "diagnoses": caption, "confidence": 80})  # Replace with actual DeepSeek call
                    return {f"{model_name}_Output": output, f"{model_name}_Version": info['version']}
                except Exception as e:
                    return {f"{model_name}_Output": f"Error: {str(e)}", f"{model_name}_Version": info['version']}

            async def process_batch():
                tasks = []
                for i, item in enumerate(batch_files):
                    row = {
                        'Image_File_Name': item['file'].name,
                        'Timestamp': datetime.now(),
                        'Metadata': json.dumps(item['metadata']) if item['metadata'] else 'N/A'
                    }
                    for model in st.session_state.selected_models:
                        tasks.append(analyze_image(item, model))
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for res in results:
                        if not isinstance(res, Exception):
                            row.update(res)
                        else:
                            row.update({f"{model}_Output": f"Error: {str(res)}"})
                    st.session_state.results_df = pd.concat([st.session_state.results_df, pd.DataFrame([row])], ignore_index=True)
                    progress.progress((i + 1) / len(batch_files))
                    status_text.text(f"Processed {i + 1}/{len(batch_files)} images")
                    await asyncio.sleep(0.1)  # Prevent UI freeze

            asyncio.run(process_batch())
        st.success("Batch processed successfully!")
        
        # Display results
        st.dataframe(st.session_state.results_df)

        # Visualize confidence scores
        if not st.session_state.results_df.empty:
            confidence_data = []
            for model in st.session_state.selected_models:
                output_col = f"{model}_Output"
                if output_col in st.session_state.results_df:
                    confidences = []
                    for output in st.session_state.results_df[output_col]:
                        try:
                            json_output = json.loads(output)
                            confidences.append(json_output.get('confidence', 0))
                        except:
                            confidences.append(0)
                    confidence_data.append({'Model': model, 'Confidence': confidences})
            
            if confidence_data:
                fig = px.bar(
                    x=st.session_state.results_df['Image_File_Name'],
                    y=[data['Confidence'] for data in confidence_data],
                    barmode='group',
                    labels={'x': 'Image', 'y': 'Confidence (%)', 'variable': 'Model'},
                    title='Model Confidence Scores'
                )
                fig.update_layout(legend_title_text='Model')
                st.plotly_chart(fig)

        # Incremental download
        csv = st.session_state.results_df.to_csv(index=False)
        st.download_button("Download Partial Results as CSV", csv, f"mri_partial_results_{st.session_state.current_batch + 1}.csv", "text/csv")

    # Step 4: Next batch
    if end_idx < total_images:
        if st.button("Analyze Next Batch"):
            st.session_state.current_batch += 1
            st.rerun()
    else:
        st.success("All images processed!")

# Final download
if not st.session_state.results_df.empty:
    csv = st.session_state.results_df.to_csv(index=False)
    st.download_button("Download Full Results as CSV", csv, "mri_results.csv", "text/csv")
