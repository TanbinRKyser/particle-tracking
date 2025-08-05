# Particle‑Tracking

**Particle‑Tracking** is a Python‑based framework for detecting and tracing particles in video/image data. It provides tools for preprocessing, particle identification, trajectory linking, analysis, and visualization.

---

## 🚀 Features

- Support for 2D/3D video/image datasets (e.g. TIFF, AVI, MP4).
- Modular pipeline: detection → linking → refinement → analysis.
- Visualization of tracked trajectories.
- Statistical analysis: displacement, mean squared displacement (MSD), diffusion rates, dwell times, etc.
- Extensible API for integrating custom detection or linking algorithms.

---

## 📁 Repository Structure

text
/
├── detection/ # particle identification modules (e.g., LoG, thresholding)
├── linking/ # trajectory linking strategies (e.g., nearest neighbor, Kalman filter)
├── refinement/ # cleaning, gap‑filling, stitching, interpolation
├── analysis/ # compute MSD, dwell times, diffusion coefficients
├── visualization/ # plotting & rendering trajectories
├── data/ # optional test or example data
├── notebooks/ # example usage & tutorials
├── setup.py / pyproject.toml
└── README.md # this file
