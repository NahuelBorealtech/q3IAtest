import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64  # MNIST images are 28x28, we will resize them to 64x64
CHANNELS_IMG = 1
NUM_CLASSES = 10
LATENT_DIM = 100
EMBEDDING_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 32
FEATURES_GEN = 32

# Data loading and transformation
transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# Download MNIST dataset
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model Architecture ---

class Discriminator(nn.Module):
    """
    Discriminator model for a Conditional DCGAN.
    It takes a 64x64 image and a label, and outputs a single value
    representing the probability that the image is real for the given label.
    """
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # Input: N x (channels_img + 1) x 64 x 64
            nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # Output: N x features_d*8 x 4 x 4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # nn.Sigmoid(), -- We use BCEWithLogitsLoss
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        # Reshape label embedding to be concatenated with the image
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1) # N x (C+1) x H x W
        return self.disc(x)


class Generator(nn.Module):
    """
    Generator model for a Conditional DCGAN.
    It takes a latent vector (noise) and a label, and outputs a 64x64 image.
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

def weights_init(m):
    """
    Initializes weights according to the DCGAN paper.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# --- Initialization ---

# Initialize models
gen = Generator(LATENT_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, EMBEDDING_DIM).to(DEVICE)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE).to(DEVICE)

# Apply weight initialization
gen.apply(weights_init)
disc.apply(weights_init)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCEWithLogitsLoss()

# --- Training Loop ---

print("Starting Training...")
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # --- Train Discriminator ---
        # Loss on real images
        disc.zero_grad()
        disc_real_pred = disc(real, labels).reshape(-1)
        loss_disc_real = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        
        # Loss on fake images
        noise = torch.randn(real.size(0), LATENT_DIM, 1, 1).to(DEVICE)
        fake = gen(noise, labels)
        disc_fake_pred = disc(fake.detach(), labels).reshape(-1)
        loss_disc_fake = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        
        # Total discriminator loss
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        loss_disc.backward()
        opt_disc.step()

        # --- Train Generator ---
        # We want the discriminator to classify the fake images as real
        gen.zero_grad()
        output = disc(fake, labels).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        loss_gen.backward()
        opt_gen.step()

        # Print losses
        if batch_idx % 200 == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

print("Training finished.")

# --- Save the Generator ---
print("Saving generator model to generator.pth...")
torch.save(gen.state_dict(), "generator.pth")
print("Model saved.") 