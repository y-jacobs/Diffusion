# Simple Diffusion Model

A basic implementation of a diffusion model for image generation.

Diffusion models generate images by learning to reverse a noise process. The model is trained on real images that have been gradually corrupted with noise until they become pure random noise. During training, a UNet neural network learns to predict and remove this noise step by step. Once trained, the model can generate new images by starting with random noise and repeatedly applying the learned denoising process to gradually reveal a coherent image.