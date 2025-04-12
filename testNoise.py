import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2, snoise2  # requires the 'noise' package (pip install noise)
from scipy.ndimage import gaussian_filter  # for smoothing value noise

# Grid dimensions and noise parameters
size = 256      # size of the noise map (256x256)
scale = 100.0   # scaling factor: higher values mean lower frequency (larger features)
octaves = 6     # number of noise layers for fBm
persistence = 0.5
lacunarity = 2.0

# Initialize empty arrays for each noise type
perlin_noise = np.zeros((size, size))
fbm_noise = np.zeros((size, size))
simplex_noise = np.zeros((size, size))

# Generate basic Perlin noise (using a single octave)
for i in range(size):
    for j in range(size):
        x = i / scale
        y = j / scale
        # using one octave gives a smooth, low-frequency map
        perlin_noise[i, j] = pnoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0)

# Generate fractal (fBm) noise using Perlin noise (multiple octaves)
for i in range(size):
    for j in range(size):
        x = i / scale
        y = j / scale
        fbm_noise[i, j] = pnoise2(x, y, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

# Generate Simplex noise (with multiple octaves)
for i in range(size):
    for j in range(size):
        x = i / scale
        y = j / scale
        simplex_noise[i, j] = snoise2(x, y, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

# Generate value noise by creating a grid of random values and then smoothing it
value_noise = np.random.rand(size, size)
# Smoothing using a Gaussian filter (the sigma value can be adjusted to change the smoothness)
value_noise = gaussian_filter(value_noise, sigma=scale/50.0)

# Plot the noise maps
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(perlin_noise, cmap='gray')
axes[0, 0].set_title("Basic Perlin Noise")
axes[0, 0].axis("off")

axes[0, 1].imshow(fbm_noise, cmap='gray')
axes[0, 1].set_title("Fractal (fBm) Noise")
axes[0, 1].axis("off")

axes[1, 0].imshow(simplex_noise, cmap='gray')
axes[1, 0].set_title("Simplex Noise (fBm)")
axes[1, 0].axis("off")

axes[1, 1].imshow(value_noise, cmap='gray')
axes[1, 1].set_title("Value Noise (Smoothed)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()
