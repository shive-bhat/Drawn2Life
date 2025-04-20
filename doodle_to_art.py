import streamlit as st
from PIL import Image, ImageFilter
import torch
import torch_directml
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from datetime import datetime
import time
import os
import gc

# Configure DirectML environment variables
os.environ["DML_VISIBLE_DEVICES"] = "0"
os.environ["DML_GRAPH_CAPTURE_FORCE_SYNCHRONOUS"] = "1"
os.environ["PYTORCH_DIRECTML_CATCH_EXCEPTIONS"] = "1"
os.environ["DML_FLUSH_EVERY_SUBMISSION"] = "0"

# Configure DirectML device
device_index = 0
dml = torch_directml.device(device_index)
torch_dtype = torch.float16

# Configure Streamlit app
st.set_page_config(
    page_title="WanderInk Studio",
    page_icon="🖌️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark Theme Styling
st.markdown("""
<style>
:root {
    --primary: #2A5C7D;
    --secondary: #7EA8BE;
    --background: #1e1e1e;
    --sidebar: #2d3748;
    --text: #e2e8f0;
    --border: #4a5568;
}

.stApp {
    background: var(--background);
}

.sidebar .sidebar-content {
    background: var(--sidebar) !important;
    color: var(--text) !important;
}

.stButton>button {
    background: var(--primary) !important;
    color: white !important;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s;
}

.stButton>button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

h1 {
    color: var(--text) !important;
    border-bottom: 2px solid var(--primary);
    padding-bottom: 0.5rem;
}

.stFileUploader>div>section {
    border: 2px dashed var(--border) !important;
    border-radius: 8px;
    background: var(--sidebar) !important;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Upload interface
if not st.session_state.uploaded:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1>🖼 Wanderlink Studio</h1>
            <p style="font-size: 1.2rem; color: #a0aec0;">
                Transform sketches into digital masterpieces
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Original Streamlit file uploader
        uploaded_file = st.file_uploader(
            "Upload your sketch (PNG/JPG, max 20MB)",
            type=["png", "jpg", "jpeg"],
            key="main_uploader"
        )
        
        if uploaded_file:
            st.session_state.uploaded = True
            st.session_state.uploaded_file = uploaded_file
            st.rerun()
    
    st.stop()

# Resolution options
resolution_options = {
    "512x512": (512, 512),
    "768x768": (768, 768),
    "1080p (1920x1080)": (1920, 1080)
}

# Main application after upload
@st.cache_resource
def load_models():
    start_time = time.time()
    try:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_lineart",
            torch_dtype=torch_dtype
        ).to(dml)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None
        ).to(dml)

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_vae_slicing()
        pipe.enable_attention_slicing("max")
        return pipe
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def main():
    uploaded_file = st.session_state.uploaded_file
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, use_container_width=True, caption="Original Sketch")

    with col2:
        with st.form("generation_form"):
            st.markdown("### 🎨 Art Direction")
            prompt = st.text_input("Describe your artwork", "A professional digital painting of")
            negative_prompt = st.text_input("Exclude elements", "blurry, cartoon, text, watermark")

            with st.expander("⚙️ Advanced Settings", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    steps = st.slider("Quality Steps", 15, 50, 25)
                    strength = st.slider("Structure Fidelity", 0.5, 1.0, 0.8)
                with col_b:
                    seed = st.number_input("Random Seed", value=42)
                    resolution = st.selectbox("Resolution", list(resolution_options.keys()))
                    
                    if resolution == "1080p (1920x1080)":
                        st.warning("Requires 8GB+ VRAM. May be slow!")

            if st.form_submit_button("🖌 Generate Artwork", use_container_width=True):
                with st.spinner("✨ Creating your masterpiece..."):
                    try:
                        pipe = load_models()
                        width, height = resolution_options[resolution]
                        
                        if resolution == "1080p (1920x1080)":
                            pipe.enable_vae_tiling()
                            torch_directml.set_memory_limit(8192)

                        resized_image = original_image.resize((width, height), Image.LANCZOS)
                        
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=resized_image,
                            num_inference_steps=steps,
                            controlnet_conditioning_scale=strength,
                            generator=torch.Generator().manual_seed(seed),
                            width=width,
                            height=height
                        ).images[0]

                        result = result.filter(ImageFilter.SHARPEN)
                        if resolution == "1080p (1920x1080)":
                            result = result.filter(ImageFilter.DETAIL)

                        st.image(result, use_container_width=True, caption="Generated Artwork")
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"artwork_{timestamp}.png"
                        result.save(output_path)
                        
                        with open(output_path, "rb") as f:
                            st.download_button(
                                "📥 Download Artwork",
                                f,
                                file_name=output_path,
                                mime="image/png"
                            )

                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"**Device**: {torch_directml.device_name(device_index)}")
    st.markdown("---")
    st.markdown("### 📈 Performance")
    if st.button("🧹 Clear Memory", use_container_width=True):
        gc.collect()
        torch.cuda.empty_cache()
        st.success("Memory cleaned!")
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7EA8BE;">
        <p>Wanderlink Studio v1.0</p>
        <p>Powered by AMD DirectML</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()