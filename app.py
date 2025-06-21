import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

# --- Model Architecture ---
# The Generator class must be defined in the app script to be used by Streamlit.
# This should be the same architecture as the one used for training in train.py.

class Generator(nn.Module):
    """
    Generator model for a Conditional DCGAN.
    It takes a latent vector (noise) and a label, and outputs a 64x64 image.
    This definition MUST match the one in train.py.
    """
    def __init__(self, latent_dim, channels_img, features_g, num_classes, embedding_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x (latent_dim + embedding_dim) x 1 x 1
            self._block(latent_dim + embedding_dim, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(), # Normalize images to [-1, 1]
        )
        self.embed = nn.Embedding(num_classes, embedding_dim)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # Latent vector z: N x latent_dim x 1 x 1
        # Label embedding: N x embedding_dim
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3) # N x embed_dim x 1 x 1
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)

# --- Hyperparameters (must match training) ---
LATENT_DIM = 100
CHANNELS_IMG = 1
NUM_CLASSES = 10
EMBEDDING_DIM = 100
FEATURES_GEN = 32
MODEL_PATH = "generator.pth"

# --- Load Model ---
@st.cache_resource
def load_model():
    """
    Loads the trained generator model from the specified path.
    The model is loaded onto the CPU.
    """
    model = Generator(LATENT_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, EMBEDDING_DIM)
    try:
        # Load the state dictionary. The map_location argument ensures that the model
        # is loaded onto the CPU, which is important for deployment environments
        # that may not have a GPU.
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode
        return model
    except FileNotFoundError:
        return None

generator = load_model()

# --- Utility Function ---
def generate_images(digit, n_images=5):
    """
    Generates a specified number of images for a given digit.
    
    Args:
        digit (int): The digit (0-9) to generate.
        n_images (int): The number of images to generate.

    Returns:
        PIL.Image: A single image containing a grid of the generated digits.
    """
    if generator is None:
        return None
        
    with torch.no_grad():
        # Create latent noise vectors
        noise = torch.randn(n_images, LATENT_DIM, 1, 1, device='cpu')
        # Create corresponding labels
        labels = torch.full((n_images,), digit, device='cpu')
        
        # Generate images
        fake_images = generator(noise, labels)
        
        # Post-process for display: scale from [-1, 1] to [0, 1]
        fake_images = (fake_images * 0.5) + 0.5
        
        # Create a grid of images
        grid = make_grid(fake_images, nrow=n_images, normalize=True)
        
        # Convert to a PIL Image
        pil_img = transforms.ToPILImage()(grid)
        return pil_img

# --- Streamlit Web App UI ---

st.set_page_config(layout="wide")

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using a trained DCGAN model.")

if generator is None:
    st.error(
        f"**Model not found.** Please make sure the trained model file (`{MODEL_PATH}`) "
        "is in the same directory as this script. "
        "Follow the instructions in `README.md` to train and save the model."
    )
else:
    # User input
    digit_to_generate = st.selectbox(
        'Choose a digit to generate (0-9)',
        options=list(range(10)),
        index=2 # Default to '2'
    )

    if st.button(f"Generate images of digit {digit_to_generate}"):
        st.subheader(f"Generated images of digit {digit_to_generate}")
        
        with st.spinner("Generating..."):
            # Generate and display images
            generated_image_grid = generate_images(digit_to_generate, n_images=5)
            
            if generated_image_grid:
                st.image(generated_image_grid, use_column_width=True)
                
                # Add captions for clarity
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    col.caption(f"Sample {i+1}") 