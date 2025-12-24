## Conditional Pixel Art GAN (16x16)

A specialized Generative Adversarial Network (GAN) designed to generate high-quality 16x16 pixel art sprites conditioned on specific classes (e.g., Dragons, Knights, Potions).

This project implements a hybrid architecture combining DCGAN, ACGAN, and Spectral Normalization to solve common challenges in low-resolution image synthesis, specifically tackling Mode Collapse.

### Key Features

Conditional Generation: Users can specify exactly which class to generate (e.g., "Generate 5 Dragons").

16x16 Optimization: Uses a custom "Resize-Convolution" upsampling architecture to prevent checkerboard artifacts common in standard GANs.

Mode Collapse Solution: Implements Minibatch Standard Deviation and a custom Diversity Regularization Loss to force the generator to produce varied outputs.

Stable Training: Uses Spectral Normalization and Hinge Loss to balance the Generator and Discriminator.

Interactive Web App: Includes a full-stack Streamlit application for real-time asset generation and downloading.