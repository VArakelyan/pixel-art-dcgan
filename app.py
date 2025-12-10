import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import zipfile
from generator import PixelArtGenerator

# --- Page Config ---
st.set_page_config(page_title="Pixel Art Generator", page_icon="👾", layout="wide")

st.title("👾 Pixel Art GAN Generator")
st.markdown("Load your trained model and generate infinite pixel art sprites!")

# --- Sidebar: Configuration ---
st.sidebar.header("1. Model Configuration")
model_file = st.sidebar.file_uploader("Upload .pt Model File", type=["pt", "pth"])

# Hyperparams must match training!
noise_dim = st.sidebar.number_input("Noise Dimension", value=100)
num_classes = st.sidebar.number_input("Number of Classes", value=5)
embed_dim = st.sidebar.number_input("Embedding Dimension", value=50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Helper Functions ---
@st.cache_resource
def load_model(uploaded_file, n_dim, n_classes, e_dim):
    """Loads the model from the uploaded file buffer."""
    model = PixelArtGenerator(
        noise_dim=n_dim,
        num_classes=n_classes,
        class_embed_dim=e_dim
    ).to(device)

    # Load weights
    try:
        state_dict = torch.load(uploaded_file, map_location=device)
        # Handle cases where the file contains more than just state_dict (like optimizer state)
        if 'generator_state' in state_dict:
            model.load_state_dict(state_dict['generator_state'])
        else:
            model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def generate_images(model, counts_per_class):
    """Generates images based on user request."""
    generated_images = []

    for class_idx, count in counts_per_class.items():
        if count > 0:
            with torch.no_grad():
                # Generate noise
                z = torch.randn(count, model.noise_dim, device=device)
                # Create labels
                labels = torch.full((count,), class_idx, dtype=torch.long, device=device)

                # Inference
                fake_imgs = model(z, labels)

                # Post-process ([-1, 1] -> [0, 255])
                fake_imgs = fake_imgs.cpu() * 0.5 + 0.5
                fake_imgs = torch.clamp(fake_imgs, 0, 1)

                for i in range(count):
                    img_tensor = fake_imgs[i]
                    # Convert to PIL
                    ndarr = img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    im = Image.fromarray(ndarr)
                    generated_images.append((class_idx, im))

    return generated_images


# --- Main Interface ---

if model_file is not None:
    st.sidebar.success("Model uploaded!")
    model = load_model(model_file, noise_dim, num_classes, embed_dim)

    if model:
        st.header("2. Generation Settings")

        # Create columns for inputs
        cols = st.columns(3)
        counts = {}

        st.write("How many images do you want for each class?")

        # Dynamic sliders for each class
        input_cols = st.columns(min(num_classes, 5))
        for i in range(num_classes):
            col_idx = i % 5
            with input_cols[col_idx]:
                counts[i] = st.number_input(f"Class {i}", min_value=0, max_value=50, value=1, key=f"c_{i}")

        if st.button("✨ Generate Images", type="primary"):
            with st.spinner("Generating pixel art..."):
                results = generate_images(model, counts)

            if results:
                st.subheader("3. Results")

                # Display Grid
                # Adjust width for pixel art look
                st.markdown("""
                <style>
                img { image-rendering: pixelated; }
                </style>
                """, unsafe_allow_html=True)

                display_cols = st.columns(8)
                for idx, (class_id, img) in enumerate(results):
                    col = display_cols[idx % 8]
                    # Resize for better visibility in browser (pixel art is small)
                    display_img = img.resize((128, 128), resample=Image.NEAREST)
                    col.image(display_img, caption=f"Class {class_id}", use_container_width=True)

                # --- Download Zip ---
                st.subheader("4. Download")
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for i, (class_id, img) in enumerate(results):
                        # Save to buffer
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        zf.writestr(f"class_{class_id}_{i}.png", img_byte_arr.getvalue())

                st.download_button(
                    label="📥 Download All as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="generated_pixel_art.zip",
                    mime="application/zip"
                )
            else:
                st.warning("Please select at least one image to generate.")

else:
    st.info("👈 Please upload your trained .pt model file in the sidebar to get started.")
    st.markdown("### Expected Model Architecture")
    st.code(str(PixelArtGenerator()), language="python")