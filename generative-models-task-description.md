# Claude Code Prompt: Generative Models for Cat Image Generation

## Project Goal

Implement and compare three generative model families — VAE, GAN, and Diffusion Model — for cat image generation. The project is structured as three Jupyter notebooks (one per model), each importing from shared Python source files. The notebooks are the experiment interface: they define all parameters and run training. The shared source files contain all model, training, and evaluation logic.

Each notebook must have:
- A **parameters cell** where the user sets dataset config, model hyperparameters, and training config — all in one place, clearly grouped
- A **training cell** that runs the full training loop by calling into the shared source modules

---

## Dataset

**Primary dataset:** [Cat Dataset](https://www.kaggle.com/datasets/crawford/cat-dataset)

- Images have **variable resolutions** — this must be handled during preprocessing (resize + center crop to a fixed target size)
- Normalize pixel values to `[-1, 1]`
- Support train/validation split
- Optional data augmentation (e.g. random horizontal flip)

**Extension dataset:** [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats)

- Used in a later phase to train the best model on both classes
- Compare generated images: do they resemble cats, dogs, or a blend?
- Class-conditional generation is a bonus; unconditional generation on the mixed dataset is the baseline

---

## Models to Implement

### 1. Variational Autoencoder (VAE)

- Standard convolutional VAE with encoder, reparameterization, and decoder
- Loss: reconstruction loss + KL divergence, with a configurable `beta` weight (beta-VAE formulation)
- Key hyperparameters to expose: `latent_dim`, encoder/decoder channel widths, `beta`, learning rate

### 2. Generative Adversarial Network (GAN)

- DCGAN-style generator and discriminator as the baseline architecture
- Address the **mode collapse problem**: detect it (e.g. by monitoring discriminator loss variance or sample diversity) and apply mitigation strategies such as:
  - Label smoothing
  - Spectral normalization on the discriminator
  - Gradient penalty (WGAN-GP style) as an optional extension
- Key hyperparameters to expose: `latent_dim`, generator/discriminator feature map sizes, learning rates, label smoothing coefficient, number of discriminator updates per generator update

### 3. Diffusion Model (DDPM)

- Standard denoising diffusion probabilistic model
- UNet backbone with sinusoidal time embeddings
- Support both **linear** and **cosine** noise schedules
- Key hyperparameters to expose: number of timesteps, beta schedule type, UNet channel widths, learning rate

---

## Experiments Required by the Project

### Hyperparameter Investigation
Each notebook should make it easy to re-run training with different configs. The parameters cell is the single place to change things. Run at least a few variants per model (e.g. different `latent_dim`, different `beta` for VAE; different architectures or schedule for diffusion).

### Quantitative Comparison — FID
- Compute **Fréchet Inception Distance (FID)** for each trained model
- Use InceptionV3 features; compute on a representative sample of generated images vs. real images
- Cache real image statistics to avoid recomputing across runs
- Report FID scores in the notebooks after training

### Qualitative Assessment
- Save grids of generated samples periodically during training and after completion
- Visualize training curves (losses over epochs)

### Latent Space Interpolation
This is a required deliverable for the best-performing model:
1. Generate two images and save their latent noise vectors (`z1`, `z2`)
2. Linearly interpolate between `z1` and `z2` to produce 8 intermediate latent vectors
3. Generate images from all 10 latent vectors (2 endpoints + 8 intermediate)
4. Display and save the sequence; discuss what the smooth (or non-smooth) transition reveals about the latent space

For the **diffusion model**, the "latent" is the initial noise tensor fed to the reverse process. For the **VAE**, it is the sampled `z` vector. For the **GAN**, it is the generator input noise vector.

### Mode Collapse (GAN)
- Monitor for mode collapse during GAN training
- Log a warning when collapse is detected
- Document what mitigation strategies were tried and their effect

### Cats + Dogs Extension
- Train the best model on the combined cats and dogs dataset
- Compare FID and visual quality against the cats-only results
- Discuss whether generated images are class-distinct or blended

---

## Notebook Structure

Each of the three notebooks (`01_vae.ipynb`, `02_gan.ipynb`, `03_diffusion.ipynb`) must follow this cell layout:

1. **Imports** — import from shared source modules
2. **Parameters** — single cell with all configurable values: dataset path, image size, batch size, model architecture hyperparameters, training hyperparameters (epochs, learning rate, scheduler, seed, checkpoint frequency, output directory)
3. **Setup** — instantiate dataset, model, optimizer, trainer; print parameter count; optionally resume from checkpoint
4. **Training** — run the training loop; print per-epoch metrics
5. **Evaluation** — generate samples, compute and print FID score, plot loss curves
6. **Interpolation** — run the latent interpolation experiment and display the 10-image sequence

Cells 3–6 should require no edits from the user — they read from the config defined in cell 2.

---

## Resource Constraints

Training must be feasible on a **single free-tier GPU** (Kaggle or Google Colab):

- Default `image_size` should be 64×64
- For the diffusion model, note that reducing timesteps from 1000 to 200–500 greatly speeds up training with modest quality loss
- Include approximate training time estimates as comments in the parameters cell

---

## Evaluation Summary

At the end of each notebook, report:
- FID score
- Training time
- Representative generated image grid
- Loss curves
- Interpolation results (for the best model)

The final comparison across all three models should summarize FID scores and qualitative observations side by side.
