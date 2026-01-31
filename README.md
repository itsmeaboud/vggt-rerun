<div align="left">

# VGGT Rerun Visualizer

**Interactive 3D Visualization for [VGGT](https://github.com/facebookresearch/VGGT) using [Rerun](https://rerun.io/)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Rerun](https://img.shields.io/badge/Rerun-0.15+-ff69b4.svg)](https://rerun.io/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://www.gradio.app/)

<br />

![VGGT Visualization Demo](demo/demo.gif)

<br />

</div>

## Overview

**VGGT-Rerun** is a lightweight tool that bridges the gap between the [Video Generalized Geometry Transformer (VGGT)](https://github.com/facebookresearch/VGGT) and the [Rerun SDK](https://rerun.io/).

It provides a user-friendly **Gradio Web UI** to visualize **point clouds**, **camera trajectories**, and **depth maps** in real-time.

## Features

- **Web Interface:** Easy drag-and-drop UI powered by Gradio
- **Interactive 3D View:** Rotate, zoom, and inspect reconstructed scenes in Rerun
- **Camera Frustums:** Visualize the exact pose (extrinsics) and intrinsics of every frame
- **Depth Projection:** See how 2D depth maps project into 3D space

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/itsmeaboud/vggt-rerun.git
cd vggt-rerun
```

### 2. Environment Setup

We recommend using Miniconda to manage dependencies.

```bash
# Create environment
conda create -n vggt_viz python=3.10
conda activate vggt_viz

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Launch the App

Start the Gradio web interface by running:

```bash
python -m scripts.gardio_app
```

### 2. Open in Browser

Click the local URL shown in your terminal (usually `http://127.0.0.1:7860`).

### 3. Run Visualization

1. **Upload Images:** Drag and drop your images into the input box
2. **Run:** Click the "Run" button to start the VGGT model
3. **View:** The Rerun viewer will launch automatically to show the 3D results

---

## Project Structure

| File / Folder | Description |
|---|---|
| `scripts/gardio_app.py` | The main Gradio web application |
| `scripts/inference.py` | Core pipeline: Loads model and processes data |
| `scripts/visualizer.py` | Helper functions for Rerun visualization |
| `vggt/` | Original VGGT model implementation |

---

## Acknowledgements

This project builds upon the excellent work of:

- **VGGT:** Video Generalized Geometry Transformer ([Facebook Research](https://github.com/facebookresearch/VGGT))
- **Rerun:** The visualization SDK used for 3D rendering ([rerun.io](https://rerun.io/))

---

